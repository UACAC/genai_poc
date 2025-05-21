import os
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from sqlalchemy.orm import Session
from services.database import SessionLocal, ComplianceAgent, ComplianceSequence, DebateSession
from services.llm_service import LLMService

llm_service = LLMService()

class AgentService:
    """Service for managing compliance checks and debates dynamically."""

    def __init__(self):
        self.compliance_agents = []

    def load_selected_compliance_agents(self, agent_ids):
        """Load only the selected compliance agents from the database."""
        session = SessionLocal()
        try:
            self.compliance_agents = []
            agents = session.query(ComplianceAgent).filter(ComplianceAgent.id.in_(agent_ids)).all()
            for agent in agents:
                self.compliance_agents.append({
                    "id": agent.id,
                    "name": agent.name,
                    "model_name": agent.model_name.lower(),  # Normalize model names
                    "system_prompt": agent.system_prompt,
                    "user_prompt_template": agent.user_prompt_template
                })
        finally:
            session.close()

    def run_compliance_check(self, data_sample: str, agent_ids: list[int], db: Session):
        """
        1) Run parallel compliance checks using selected agents.
        2) If all are compliant, return. Otherwise, create a new debate session,
        store the agents, and run the debate automatically.
        """
        # Load the specified agents into self.compliance_agents
        self.load_selected_compliance_agents(agent_ids)

        # Step 1: Run parallel checks
        compliance_results = self.run_parallel_compliance_checks(data_sample)

        # Step 2: Determine overall compliance
        bool_vals = [res["compliant"] for res in compliance_results.values() if res["compliant"] is not None]
        all_compliant = bool_vals and all(bool_vals)

        if all_compliant:
            return {
                "overall_compliance": True,
                "details": compliance_results
            }
        else:
            session_id = str(uuid.uuid4())

            # Store these agents in the DebateSession table
            for idx, agent_info in enumerate(self.compliance_agents):
                db.add(DebateSession(
                    session_id=session_id,
                    compliance_agent_id=agent_info["id"],
                    debate_order=idx + 1
                ))
            db.commit()

            # Step 3: Run debate with the newly created session
            debate_results = self.run_debate(session_id, data_sample)

            return {
                "overall_compliance": False,
                "details": compliance_results,
                "debate_results": debate_results,
                "session_id": session_id
            }

    def run_parallel_compliance_checks(self, data_sample: str):
        """Run compliance checks in parallel."""
        results = {}
        with ThreadPoolExecutor() as executor:
            futures = {
                executor.submit(self.verify_compliance, agent, data_sample): i
                for i, agent in enumerate(self.compliance_agents)
            }
            for future in as_completed(futures):
                idx = futures[future]
                results[idx] = future.result()
        return results

    def verify_compliance(self, agent: dict, data_sample: str):
        """
        Check compliance using the specified agent.
        Expect the agent to reply with either "Yes" or "No" 
        plus an explanation.
        """
        model_name = agent["model_name"]

        if model_name == "gpt-4" or model_name == "gpt4":
            raw_text = llm_service.query_gpt4(prompt=data_sample).strip()
        elif model_name == "llama" or model_name == "llama3":
            raw_text = llm_service.query_llama(prompt=data_sample).strip()
        elif model_name == "mistral":
            raw_text = llm_service.query_mistral(prompt=data_sample).strip()
        elif model_name == "gemma":
            raw_text = llm_service.query_gemma(prompt=data_sample).strip()
        else:
            raw_text = f"Error: Model '{model_name}' not recognized."

        # Parse response: first line should be "Yes" or "No"
        lines = raw_text.split("\n", 1)
        first_line = lines[0].lower()

        if "yes" in first_line:
            compliant = True
        elif "no" in first_line:
            compliant = False
        else:
            compliant = None

        reason = lines[1].strip() if len(lines) > 1 else ""

        return {
            "compliant": compliant,
            "reason": reason,
            "raw_text": raw_text
        }

    def run_debate(self, session_id: str, data_sample: str):
        """
        Runs a debate session with selected agents.
        """
        debate_agents = self.load_debate_agents(session_id)

        results = {}
        with ThreadPoolExecutor() as executor:
            futures = {
                executor.submit(self.debate_compliance, agent, data_sample): agent["name"]
                for agent in debate_agents
            }
            for future in as_completed(futures):
                agent_name = futures[future]
                results[agent_name] = future.result()

        return results

    def load_debate_agents(self, session_id: str):
        """Fetch compliance agents for a specific debate session from the database."""
        session = SessionLocal()
        debate_agents = []
        try:
            debate_records = (
                session.query(DebateSession)
                .filter(DebateSession.session_id == session_id)
                .order_by(DebateSession.debate_order)
                .all()
            )
            for record in debate_records:
                agent = record.compliance_agent
                debate_agents.append({
                    "id": agent.id,
                    "name": agent.name,
                    "model_name": agent.model_name,
                    "system_prompt": agent.system_prompt,
                    "user_prompt_template": agent.user_prompt_template
                })
        finally:
            session.close()
        return debate_agents

    def debate_compliance(self, agent: dict, data_sample: str):
        """
        Runs compliance check during a debate session.
        """
        debate_prompt = (
            f"Agent {agent['name']} is evaluating this data again:\n"
            f"{data_sample}\n\n"
            "Do you find this compliant? Answer 'Yes' or 'No' and explain why."
        )

        model_name = agent["model_name"].lower()

        if model_name == "gpt-4" or model_name == "gpt4":
            response_text = llm_service.query_gpt4(prompt=debate_prompt)
        elif model_name == "llama" or model_name == "llama3":
            response_text = llm_service.query_llama(prompt=debate_prompt)
        elif model_name == "mistral":
            response_text = llm_service.query_mistral(prompt=debate_prompt)
        elif model_name == "gemma":
            response_text = llm_service.query_gemma(prompt=debate_prompt)
        else:
            return f"Error: Model '{model_name}' not recognized."

        return response_text.strip()

    def run_debate_sequence(self, db: Session, session_id: str | None, agent_ids: list[int], data_sample: str):
        """
        Runs a sequential debate session with multiple agents.
        """

        if not session_id:
            session_id = str(uuid.uuid4())

        # Clear prior debate records
        db.query(DebateSession).filter(DebateSession.session_id == session_id).delete()
        db.commit()

        # Insert agents in order
        for idx, agent_id in enumerate(agent_ids):
            db.add(DebateSession(
                session_id=session_id,
                compliance_agent_id=agent_id,
                debate_order=idx + 1
            ))
        db.commit()

        # Load debate agents from the database
        debate_agents = self.load_debate_agents(session_id)

        debate_chain = []
        context = f"Initial data:\n{data_sample}\n"

        for agent in debate_agents:
            agent_response = self.debate_compliance(agent, context)

            debate_chain.append({
                "agent_id": agent["id"],
                "agent_name": agent["name"],
                "response": agent_response
            })

            context += f"\n---\nAgent {agent['name']} responded:\n{agent_response}\n"

        return session_id, debate_chain