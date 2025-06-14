import streamlit as st
import requests
import os
import nest_asyncio
import datetime
import time
from utils import fetch_collections, get_available_models
import torch

torch.classes.__path__ = []
nest_asyncio.apply()

# THIS MUST BE THE VERY FIRST STREAMLIT COMMAND
st.set_page_config(page_title="AI Assistant", layout="wide")

# FastAPI API endpoints
FASTAPI_URL = os.getenv("FASTAPI_URL", "http://localhost:9020")
CHAT_ENDPOINT = f"{FASTAPI_URL}/chat"
HISTORY_ENDPOINT = f"{FASTAPI_URL}/chat-history"
HEALTH_ENDPOINT = f"{FASTAPI_URL}/health"

st.title("AI Assistant")

# Initialize session state
if 'health_status' not in st.session_state:
    st.session_state.health_status = None
if 'available_models' not in st.session_state:
    st.session_state.available_models = []
if 'collections' not in st.session_state:
    st.session_state.collections = []
if 'agents_data' not in st.session_state:
    st.session_state.agents_data = []
if 'debate_sequence' not in st.session_state:
    st.session_state.debate_sequence = []

# Cache functions
@st.cache_data(ttl=300)
def get_available_models_cached():
    return get_available_models()

def check_model_status(model_name):
    """Check if a specific model is loaded in Ollama"""
    try:
        response = requests.get(f"{FASTAPI_URL}/health", timeout=5)
        if response.ok:
            health_data = response.json()
            models = health_data.get("models", {})
            return models.get(model_name, "unknown")
    except:
        return "unknown"

# Model configurations
model_key_map = {
    "GPT-4": "gpt-4",
    "GPT-3.5 Turbo": "gpt-3.5-turbo", 
    "LLaMA": "llama3",
}

model_descriptions = {
    "gpt-4": "🧠 Most capable model for complex analysis",
    "gpt-3.5-turbo": "💡 Cost-effective model for general tasks",
    "llama3": "🦙 Fast and efficient general-purpose model",
}

# ----------------------------------------------------------------------
# SIDEBAR - SYSTEM STATUS & CONTROLS
# ----------------------------------------------------------------------
with st.sidebar:
    st.header("System Status")
    
    # Health check
    if st.button("Check Health"):
        try:
            with st.spinner("Checking health..."):
                response = requests.get(HEALTH_ENDPOINT, timeout=10)
                if response.ok:
                    st.session_state.health_status = response.json()
                    st.success("Online")
                else:
                    st.error("System Issues")
        except Exception as e:
            st.error(f"Cannot connect to API: {e}")
    
    # Display cached health status
    if st.session_state.health_status:
        with st.expander("System Details"):
            st.json(st.session_state.health_status)

    # st.header("Available Models")
    
    # # Model management buttons
    # col1, col2 = st.columns(2)
    # with col1:
    #     if st.button("Refresh"):
    #         st.session_state.available_models = get_available_models_cached()
    #         st.session_state.legal_models = []
    #         st.success("Refreshed!")
    
    # # with col2:
    # #     if st.button("🧪 Test"):
    # #         with st.spinner("Testing..."):
    # #             available_models = check_model_availability()
    # #             st.session_state.available_models = available_models
    # #             st.success(f"Found {len(available_models)} models")
    
    
    # # Display available models
    # available_models = st.session_state.available_models or get_available_models_cached()

    # if available_models:
    #     for model in available_models:
    #         if model in model_descriptions:
    #             st.text(f"{model}")
    #         else:
    #             st.text(f"{model}")
    # else:
    #     st.info("Click 'Refresh' to load models")

    st.header("Collections")
    
    if st.button("Load Collections"):
        try:
            st.session_state.collections = fetch_collections()
            st.success("Collections loaded!")
        except Exception as e:
            st.error(f"Error: {e}")
    
    # Display collections
    collections = st.session_state.collections
    if collections:
        for collection in collections:
            st.text(f"{collection}")
    else:
        st.info("Click 'Load Collections' to see available databases")

# Get collections for main interface
try:
    if not st.session_state.collections:
        collections = fetch_collections()
        st.session_state.collections = collections
    else:
        collections = st.session_state.collections
except:
    collections = []

# ----------------------------------------------------------------------
# MAIN INTERFACE
# ----------------------------------------------------------------------
# Chat mode selection
# Update your main chat mode selection to include the new mode:
chat_mode = st.radio(
    "Select Mode:",
    ["💬 Direct Chat", "🤖 AI Agent Simulation", "🛠️ Create Agent", "📊 Session History"],
    horizontal=True
)


# ----------------------------------------------------------------------
# DIRECT CHAT MODE
# ----------------------------------------------------------------------
if chat_mode == "💬 Direct Chat":
    st.markdown("---")
    
    # Model selection (common for all modes)
    col1, col2 = st.columns([2, 1])
    with col1:
        mode = st.selectbox("Select AI Model:", list(model_key_map.keys()))
        if model_key_map[mode] in model_descriptions:
            st.info(model_descriptions[model_key_map[mode]])
            
    use_rag = st.checkbox("Use RAG")
    if use_rag and collections:
        collection_name = st.selectbox("Document Collection:", collections)
    else:
        collection_name = None
    
    user_input = st.text_area(
        "Ask your question:", 
        height=100, 
        placeholder="Example: Analyze this to determine if it is within standards..."
    )

    if st.button("Get Analysis", type="primary"):
        if not user_input:
            st.warning("Please enter a question.")
        elif use_rag and not collection_name:
            st.error("Please select a collection for RAG mode.")
        else:
            payload = {
                "query": user_input,
                "model": model_key_map[mode],
                "use_rag": use_rag,
                "collection_name": collection_name if use_rag else None
            }

            with st.spinner(f"{mode} is analyzing your question..."):
                status_placeholder = st.empty()
                
                try:
                    status_placeholder.info("Connecting to AI model...")
                    
                    response = requests.post(CHAT_ENDPOINT, json=payload, timeout=300)  
                    
                    if response.ok:
                        result = response.json().get("response", "")
                        status_placeholder.empty()
                        
                        st.success("Analysis Complete:")
                        st.markdown("### Analysis Results:")
                        st.markdown(result)
                        
                        if "response_time_ms" in response.json():
                            response_time = response.json()["response_time_ms"]
                            st.caption(f"Response time: {response_time/1000:.2f} seconds")
                            
                    else:
                        status_placeholder.empty()
                        error_detail = response.json().get("detail", response.text) if response.headers.get("content-type") == "application/json" else response.text
                        
                        if "model" in error_detail and "not found" in error_detail:
                            st.error("Model is loading for the first time. This may take 1-2 minutes. Please try again.")
                            st.info("Tip: The first request to each model takes longer as it loads into memory.")
                        else:
                            st.error(f"Error {response.status_code}: {error_detail}")
                            
                except requests.exceptions.Timeout:
                    status_placeholder.empty()
                    st.error("Request timed out. The model might be loading - please try again in a moment.")
                    st.info("Large models can take 1-2 minutes to load on first use.")
                except requests.exceptions.RequestException as e:
                    status_placeholder.empty()
                    st.error(f"Request failed: {e}")
                    
    st.markdown("---")
    st.header("Chat History")
    if st.button("Load Chat History"):
        try:
            with st.spinner("Loading chat history..."):
                response = requests.get(HISTORY_ENDPOINT, timeout=10)
                if response.ok:
                    history = response.json()
                    if not history:
                        st.info("No chat history found.")
                    else:
                        for record in reversed(history[-10:]):
                            with st.expander(f"💬 {record['timestamp'][:19]}"):
                                st.markdown(f"**User:** {record['user_query']}")
                                st.markdown(f"**Response:** {record['response']}")
                else:
                    st.error(f"Failed to fetch history: {response.status_code}")
        except requests.exceptions.RequestException as e:
            st.error(f"Request failed: {e}")
        

# ----------------------------------------------------------------------
# AI AGENT SIMULATION MODE
# ----------------------------------------------------------------------
elif chat_mode == "🤖 AI Agent Simulation":
    st.markdown("---")
    
    # Load agents with enhanced error handling
    col1_load, col2_load = st.columns([1, 2])
    
    with col1_load:
        if st.button("Refresh Agent List"):
            try:
                with st.spinner("Loading agents from database..."):
                    agents_response = requests.get(f"{FASTAPI_URL}/get-agents", timeout=10)
                    if agents_response.status_code == 200:
                        agent_data = agents_response.json()
                        st.session_state.agents_data = agent_data.get("agents", [])
                        st.success(f"Loaded {len(st.session_state.agents_data)} agents")
                    else:
                        st.warning(f"Could not load agents (Status: {agents_response.status_code})")
                        st.error(f"Response: {agents_response.text}")
            except Exception as e:
                st.error(f"Error loading agents: {e}")
    
    with col2_load:
        if st.session_state.agents_data:
            total_agents = len(st.session_state.agents_data)
            active_agents = sum(1 for agent in st.session_state.agents_data if agent.get("is_active", True))
            st.metric("Total Agents", total_agents, delta=f"{active_agents} active")

    # Check if we have agents data
    if st.session_state.agents_data:
        agents = st.session_state.agents_data
        
        # Create agent choices dictionary for selection
        agent_choices = {f"{agent['name']} ({agent['model_name']})": agent["id"] for agent in agents}
        
        # Enhanced agents display
        agents_table_data = []
        for agent in agents:
            agents_table_data.append({
                "ID": agent.get("id", "N/A"),
                "Name": agent.get("name", "Unknown"),
                "Model": agent.get("model_name", "Unknown"),
                "Queries": agent.get("total_queries", 0),
                "Avg Response": f"{agent.get('avg_response_time_ms', 0):.0f}ms" if agent.get('avg_response_time_ms') else "N/A",
                "Success Rate": f"{agent.get('success_rate', 0)*100:.1f}%" if agent.get('success_rate') else "N/A",
                "Status": "Active" if agent.get("is_active", True) else "Inactive",
                "Created": agent.get("created_at", "Unknown")[:10] if agent.get("created_at") else "Unknown",
                "System Prompt": agent.get("system_prompt", "")[:100] + ("..." if len(agent.get("system_prompt", "")) > 100 else ""),
                "User Template": agent.get("user_prompt_template", "")[:100] + ("..." if len(agent.get("user_prompt_template", "")) > 100 else "")
            })
        
        # Display agents table
        st.dataframe(agents_table_data, use_container_width=True, height=400)
        
        # Single Agent Analysis Section
        st.subheader("Single Agent Analysis")
        
        # Analysis content
        analysis_content = st.text_area(
            "Legal Content for Agent Analysis", 
            placeholder="Paste contract text, legal documents, or content for specialized analysis...",
            height=150
        )
        
        selected_agents = st.multiselect(
            "Select Specialized Agents for Analysis", 
            list(agent_choices.keys()),
            help="Choose multiple agents to get different perspectives on your content"
        )

        # Analysis type selection
        analysis_type = st.selectbox(
            "Analysis Type:",
            ["Direct Analysis", "RAG-Enhanced Analysis"],
            help="Direct uses content as-is, RAG adds context from your knowledge base"
        )

        if analysis_type == "RAG-Enhanced Analysis":
            if collections:
                collection_name = st.selectbox("Collection to utilize Retrieval Augmented Generation:", collections)
            else:
                st.warning("No collections available for RAG analysis")
                collection_name = None
        else:
            collection_name = None

        if st.button("Run Agent Analysis", type="primary"):
            if not analysis_content or not selected_agents:
                st.warning("Please provide content and select at least one agent.")
            else:
                agent_ids = [agent_choices[name] for name in selected_agents]
                
                # Choose endpoint based on analysis type
                if analysis_type == "RAG-Enhanced Analysis" and collection_name:
                    payload = {
                        "query_text": analysis_content,
                        "collection_name": collection_name,
                        "agent_ids": agent_ids
                    }
                    endpoint = f"{FASTAPI_URL}/rag-check"
                else:
                    payload = {
                        "data_sample": analysis_content,
                        "agent_ids": agent_ids
                    }
                    endpoint = f"{FASTAPI_URL}/compliance-check"
                
                with st.spinner("🤖 Specialized agents are analyzing the content..."):
                    status_placeholder = st.empty()
                    try:
                        status_placeholder.info("Connecting to AI model...")
                        response = requests.post(endpoint, json=payload, timeout=300)
                        
                        # if response.ok:
                        #     result = response.json().get("response", "")
                        #     status_placeholder.empty()
                            
                        #     st.success("Agent Analysis Complete!")
                        #     st.markdown("### Analysis Results:")
                        #     st.markdown(result)
                            
                        #     if "response_time_ms" in response.json():
                        #         response_time = response.json()["response_time_ms"]
                        #         st.caption(f"Response time: {response_time/1000:.2f} seconds")
                        
                        # else:
                        #     status_placeholder.empty()
                        #     error_detail = response.json().get("detail", response.text) if response.headers.get("content-type") == "application/json" else response.text
                            
                        #     if "model" in error_detail and "not found" in error_detail:
                        #         st.error("Model is loading for the first time. This may take 1-2 minutes. Please try again.")
                        #         st.info("Tip: The first request to each model takes longer as it loads into memory.")
                        #     else:
                        #         st.error(f"Error {response.status_code}: {error_detail}")
                        
                        if response.status_code == 200:
                            result = response.json()
                            
                            # Display results
                            agent_responses = result.get("agent_responses", {})
                            if agent_responses:
                                for agent_name, analysis in agent_responses.items():
                                    with st.expander(f"🤖 {agent_name} Analysis", expanded=True):
                                        st.markdown(analysis)
                            else:
                                # Handle compliance check format
                                details = result.get("details", {})
                                for idx, analysis in details.items():
                                    agent_name = analysis.get("agent_name", f"Agent {idx}")
                                    reason = analysis.get("reason", analysis.get("raw_text", "No analysis"))
                                    
                                    with st.expander(f"🤖 {agent_name} Analysis", expanded=True):
                                        st.markdown(reason)
                                        
                            if "session_id" in result:
                                st.info(f"Analysis Session ID: {result['session_id']}")
                                
                        else:
                            error_detail = response.json().get("detail", response.text) if response.headers.get("content-type") == "application/json" else response.text
                            st.error(f"Error {response.status_code}: {error_detail}")
                            
                    except requests.exceptions.Timeout:
                        status_placeholder.empty()
                        st.error("Request timed out. The model might be loading - please try again in a moment.")
                        st.info("Large models can take 1-2 minutes to load on first use.")
                        # st.error("Analysis timed out. Try with shorter content or fewer agents.")
                    except Exception as e:
                        status_placeholder.empty()
                        st.error(f"Analysis failed: {str(e)}")

        st.markdown("---")
        
        # Multi-Agent Debate Sequence Section
        st.subheader("🗣️ Multi-Agent Debate Sequence")
        st.info("**How it works**: Create a sequence of agents that will debate in order. Each agent uses their pre-configured LLM and prompts.")
        
        # Initialize session state for agent sequence
        if "debate_sequence" not in st.session_state:
            st.session_state["debate_sequence"] = []
        
        # Agent sequence builder
        col1, col2 = st.columns([3, 1])
        with col1:
            new_agent_to_add = st.selectbox(
                "Add Agent to Debate Sequence", 
                ["--Select an Agent--"] + list(agent_choices.keys()), 
                key="debate_sequence_select"
            )
        with col2:
            if st.button("➕ Add to Sequence", key="add_agent_debate"):
                if new_agent_to_add != "--Select an Agent--" and new_agent_to_add not in st.session_state["debate_sequence"]:
                    st.session_state["debate_sequence"].append(new_agent_to_add)
                    st.success(f"Added {new_agent_to_add}")
                    st.rerun()
                elif new_agent_to_add in st.session_state["debate_sequence"]:
                    st.warning("Agent already in sequence!")
            
            if st.button("🗑️ Clear All"):
                st.session_state["debate_sequence"] = []
                st.rerun()
        
        # Display current debate sequence
        if st.session_state["debate_sequence"]:
            st.write("**Current Debate Sequence:**")
            for i, agent_name in enumerate(st.session_state["debate_sequence"], 1):
                col1, col2, col3 = st.columns([1, 4, 1])
                with col1:
                    st.write(f"**{i}.**")
                with col2:
                    # Show agent info
                    agent_data = next((agent for agent in agents if f"{agent['name']} ({agent['model_name']})" == agent_name), None)
                    if agent_data:
                        st.write(f"**{agent_data['name']}** using *{agent_data['model_name']}*")
                    else:
                        st.write(f"{agent_name}")
                with col3:
                    if st.button("❌", key=f"remove_{i}", help="Remove from sequence"):
                        st.session_state["debate_sequence"].remove(agent_name)
                        st.rerun()
            
            st.markdown("---")
        else:
            st.info("Add agents to create a debate sequence")
        
        # Content and collection selection for debate
        debate_content = st.text_area(
            "Legal Content for Multi-Agent Debate", 
            placeholder="Enter the legal content that agents will debate about...",
            height=120, 
            key="debate_content"
        )
        
        # RAG selection for debate
        use_rag_debate = st.checkbox("Use RAG for Debate Context", key="rag_debate")
        if use_rag_debate and collections:
            collection_for_debate = st.selectbox(
                "ChromaDB Collection (for RAG context):", 
                collections,
                help="Select a collection to provide context for the debate"
            )
        else:
            if use_rag_debate:
                st.warning("No collections available. Agents will debate without RAG context.")
            collection_for_debate = None
        
        # Start debate button
        if st.button("🗣️ Start Multi-Agent Debate", type="primary", key="start_debate"):
            if not debate_content:
                st.warning("Please provide content for the agents to debate.")
            elif not st.session_state["debate_sequence"]:
                st.warning("Please add at least one agent to the debate sequence.")
            else:
                # Prepare the debate
                sequence_agent_ids = []
                for agent_name in st.session_state["debate_sequence"]:
                    agent_id = agent_choices.get(agent_name)
                    if agent_id:
                        sequence_agent_ids.append(agent_id)
                
                if not sequence_agent_ids:
                    st.error("Unable to find agent IDs. Please reload agents and try again.")
                else:
                    # Show what's about to happen
                    with st.expander("🔍 Debate Setup", expanded=True):
                        st.write(f"**Content**: {debate_content[:100]}{'...' if len(debate_content) > 100 else ''}")
                        st.write(f"**Agents in sequence**: {len(sequence_agent_ids)}")
                        st.write(f"**Using RAG**: {'Yes' if use_rag_debate and collection_for_debate else 'No'}")
                        if use_rag_debate and collection_for_debate:
                            st.write(f"**Collection**: {collection_for_debate}")
                    
                    # Prepare payload based on whether we're using RAG or not
                    if use_rag_debate and collection_for_debate:
                        debate_payload = {
                            "query_text": debate_content,
                            "collection_name": collection_for_debate,
                            "agent_ids": sequence_agent_ids
                        }
                        endpoint = f"{FASTAPI_URL}/rag-debate-sequence"
                    else:
                        debate_payload = {
                            "data_sample": debate_content,
                            "agent_ids": sequence_agent_ids
                        }
                        endpoint = f"{FASTAPI_URL}/compliance-check"  # Use compliance check for non-RAG debate
                    
                    # Start the debate
                    with st.spinner(f"🗣️ {len(sequence_agent_ids)} agents are debating..."):
                        status_placeholder = st.empty()
                        try:
                            status_placeholder.info("Connecting to debate service...")
                            response = requests.post(endpoint, json=debate_payload, timeout=300)
                            
                            if response.status_code == 200:
                                result = response.json()
                                st.success("Multi-Agent Debate Complete!")
                                
                                # Display results
                                session_id = result.get("session_id")
                                if session_id:
                                    st.info(f"Debate Session ID: `{session_id}`")
                                
                                # Show debate results
                                if "debate_chain" in result:
                                    debate_chain = result["debate_chain"]
                                    st.subheader("🗣️ Debate Sequence Results")
                                    
                                    for i, round_result in enumerate(debate_chain, 1):
                                        agent_name = round_result.get('agent_name', 'Unknown Agent')
                                        response_text = round_result.get("response", "No response")
                                        
                                        with st.expander(f"Round {i}: {agent_name}", expanded=i<=2):
                                            st.markdown(response_text)
                                            
                                            # Show metadata if available
                                            if "agent_id" in round_result:
                                                st.caption(f"Agent ID: {round_result['agent_id']}")
                                
                                elif "details" in result:
                                    # Handle compliance check format
                                    st.subheader("🗣️ Agent Analysis Results")
                                    details = result["details"]
                                    
                                    for idx, analysis in details.items():
                                        agent_name = analysis.get("agent_name", f"Agent {idx}")
                                        reason = analysis.get("reason", analysis.get("raw_text", "No analysis"))
                                        
                                        with st.expander(f"{agent_name}", expanded=True):
                                            st.markdown(reason)
                                
                                elif "agent_responses" in result:
                                    # Handle agent_responses format
                                    st.subheader("🗣️ Agent Analysis Results")
                                    agent_responses = result["agent_responses"]
                                    
                                    for agent_name, response_text in agent_responses.items():
                                        with st.expander(f"🤖 {agent_name}", expanded=True):
                                            st.markdown(response_text)
                                
                                else:
                                    st.json(result)  # Fallback to show raw result
                                    
                            else:
                                error_detail = response.json().get("detail", response.text) if response.headers.get("content-type") == "application/json" else response.text
                                st.error(f"Error {response.status_code}: {error_detail}")
                                
                        except requests.exceptions.Timeout:
                            status_placeholder.empty()
                            st.error("Request timed out. The model might be loading - please try again in a moment.")
                            st.info("Large models can take 1-2 minutes to load on first use.")
                        except Exception as e:
                            status_placeholder.empty()
                            st.error(f"Debate failed: {str(e)}")

        # Help section
        with st.expander("How Multi-Agent Debate Works"):
            st.markdown("""
            **Multi-Agent Debate Process:**
            
            1. **Create Sequence**: Add agents in the order you want them to participate
            2. **Each agent uses their own configured LLM** (LLama3)
            3. **Each agent applies their specialized prompts** (Systems Engineering analysis, Quality Control Engineer, etc.)
            4. **Sequential Analysis**: Agents review the content in order
            5. **Optional RAG**: Choose a collection to provide additional context
            
            **Example Sequence:**
            - Risk Analyzer (Llama3) → Identifies risks
            - Systems Engineer (LLama3) → Reviews requirements
            - Test Engineer (Llama3) → Provides test context
            
            **Why this is better than selecting models again:**
            - Agents already have optimized LLM+prompt combinations
            - Each agent brings their specialized expertise
            - No need to duplicate model selection
            """)

    else:
        # No agents available
        st.info("Click 'Refresh Agent List' to load your specialized legal agents")
        st.markdown("""
        **No agents found!** 
        
        To use Agent Simulation mode:
        1. Go to the '🛠️ Create Agent' tab
        2. Create some specialized legal agents
        3. Come back here to simulate multi-agent analysis
        """)

    # Footer
    st.markdown("---")
    st.markdown("* All agents provide analysis for informational purposes only*")

# ----------------------------------------------------------------------
# CREATE AGENT MODE (WITH MANAGEMENT SUB-MODES)
# ----------------------------------------------------------------------
elif chat_mode == "🛠️ Create Agent":
    st.markdown("---")
    st.header("🛠️ Agent Management")
    
    # Agent management sub-modes
    agent_mode = st.radio(
        "Select Action:",
        ["Create New Agent", "Manage Existing Agents"],
        horizontal=True
    )
    
    # ----------------------------------------------------------------------
    # CREATE NEW AGENT SUB-MODE
    # ----------------------------------------------------------------------
    if agent_mode == "Create New Agent":
        st.subheader("Create a New Agent")
        
        # Enhanced agent templates (keeping your existing templates)
        enhanced_templates = {
            "Systems Engineering Agent": {
                "description": "Systems engineering specialist applying SEBoK (Systems Engineering Body of Knowledge) principles to review and enhance requirement development processes",
                "system_prompt": """You are an experienced systems engineer with 15+ years of experience in complex system development across aerospace, defense, and technology sectors. Your expertise spans the full systems engineering lifecycle with deep knowledge of SEBoK principles. Your role is to:

                1. **Requirements Analysis**: Evaluate requirements for completeness, consistency, clarity, and traceability
                2. **SEBoK Application**: Apply Systems Engineering Body of Knowledge best practices and standards
                3. **Stakeholder Assessment**: Analyze stakeholder needs translation into verifiable requirements
                4. **Architecture Alignment**: Ensure requirements support system architecture and design decisions
                5. **Verification Planning**: Assess testability and verification approaches for each requirement
                6. **Risk Identification**: Identify requirement-related risks and mitigation strategies
                7. **Process Improvement**: Recommend enhancements to requirement development processes

                **SEBoK Framework Application:**
                - Apply life cycle processes (ISO/IEC/IEEE 15288)
                - Utilize requirements engineering best practices
                - Ensure stakeholder needs are properly captured and managed
                - Maintain traceability throughout system hierarchy
                - Consider system context and boundary definitions
                - Apply appropriate requirement attributes and characteristics

                **Technical Approach:**
                - Evaluate requirements against SMART criteria (Specific, Measurable, Achievable, Relevant, Time-bound)
                - Assess requirement quality attributes (unambiguous, complete, consistent, testable)
                - Review for proper categorization (functional, performance, interface, constraint)
                - Analyze requirement dependencies and interactions
                - Consider system integration and interface requirements
                - Evaluate for proper abstraction levels across system hierarchy""",

                "user_prompt": """As a systems engineering specialist, review the following requirement development artifact and provide a comprehensive SEBoK-based assessment:

                {data_sample}

                Please provide a detailed systems engineering analysis:

                1. **Requirements Quality Assessment** (completeness, clarity, consistency, traceability evaluation)
                2. **SEBoK Compliance Review** (alignment with systems engineering best practices and standards)
                3. **Stakeholder Analysis** (adequacy of stakeholder need capture and translation)
                4. **Architecture Considerations** (requirement support for system design and interfaces)
                5. **Verification & Validation Planning** (testability assessment and V&V approach recommendations)
                6. **Risk Analysis** (requirement-related risks and technical concerns)
                7. **Process Recommendations** (improvements to requirement development methodology)
                8. **Traceability Assessment** (requirement hierarchy and dependency analysis)
                9. **Next Steps** (prioritized actions to enhance requirement quality and process)

                Focus on applying rigorous systems engineering principles while providing practical, actionable recommendations for improvement.""",

                "temperature": 0.3,
                "max_tokens": 2000
            },
            
            "Test Engineering Agent": {
                "description": "Expert system test engineer specializing in comprehensive test case development, test plan review, and verification strategy across complex integrated systems",
                "system_prompt": """You are a senior system test engineer with 12+ years of experience in developing and executing test strategies for complex integrated systems across aerospace, automotive, telecommunications, and enterprise software domains. Your expertise encompasses the full testing lifecycle from planning through execution and reporting. Your role is to:

                1. **Test Strategy Development**: Design comprehensive test approaches covering functional, non-functional, and integration testing
                2. **Test Case Design**: Create detailed, traceable test cases using systematic design techniques (equivalence partitioning, boundary value analysis, state-based testing)
                3. **Test Plan Review**: Evaluate test plans for completeness, feasibility, risk coverage, and alignment with requirements
                4. **Verification Planning**: Develop verification strategies that demonstrate system compliance with specifications
                5. **Test Environment Design**: Specify test infrastructure, tooling, and data requirements
                6. **Risk-Based Testing**: Prioritize testing efforts based on system criticality and failure impact analysis
                7. **Test Automation Strategy**: Identify automation opportunities and develop sustainable test frameworks

                **Testing Framework Application:**
                - Apply IEEE 829 test documentation standards
                - Utilize ISO/IEC/IEEE 29119 testing principles and processes
                - Implement systematic test design techniques and coverage criteria
                - Ensure bidirectional traceability between requirements and test cases
                - Apply risk-based testing methodologies (ISO 31000)
                - Consider system integration testing approaches (big-bang, incremental, sandwich)
                - Evaluate test completion criteria and exit conditions

                **Technical Approach:**
                - Design test cases covering positive, negative, and boundary conditions
                - Develop test scenarios for system-level behaviors and emergent properties
                - Create test procedures with clear setup, execution, and validation steps
                - Specify test data requirements and management strategies
                - Define test environment configurations and dependencies
                - Plan for defect management and regression testing cycles
                - Consider performance, security, usability, and reliability testing aspects""",
                    
                    "user_prompt": """As a senior system test engineer, analyze the following testing artifact and provide a comprehensive assessment:

                {data_sample}

                Please provide a detailed system testing analysis:

                1. **Test Strategy Assessment** (completeness of testing approach and methodology)
                2. **Test Case Quality Review** (clarity, traceability, executability, and coverage evaluation)
                3. **Requirements Coverage Analysis** (mapping between requirements and test cases, gap identification)
                4. **Test Plan Evaluation** (feasibility, resource allocation, timeline, and risk considerations)
                5. **Test Environment & Infrastructure** (adequacy of test setup, tooling, and data requirements)
                6. **Integration Testing Strategy** (system integration approach and interface testing coverage)
                7. **Non-Functional Testing Coverage** (performance, security, reliability, usability considerations)
                8. **Risk Analysis & Mitigation** (testing risks and contingency planning)
                9. **Test Automation Opportunities** (automation feasibility and ROI assessment)
                10. **Process Recommendations** (improvements to testing methodology and practices)
                11. **Next Steps** (prioritized actions to enhance test quality and execution)

                Focus on providing practical, actionable recommendations that improve test effectiveness, efficiency, and system quality assurance.""",
                    
                    "temperature": 0.2,
                    "max_tokens": 2200
            },
            "Quality Engineering Agent": {
                "description": "Expert quality engineering specialist focused on implementing comprehensive quality management systems, process improvement, and quality assurance across the entire product lifecycle",
                "system_prompt": """You are a senior quality engineering professional with 15+ years of experience implementing quality management systems across manufacturing, software development, aerospace, medical devices, and automotive industries. Your expertise spans quality planning, process control, continuous improvement, and regulatory compliance. Your role is to:

                1. **Quality Management Systems**: Design and implement QMS frameworks (ISO 9001, AS9100, ISO 13485, IATF 16949)
                2. **Process Quality Control**: Establish statistical process control, quality metrics, and performance monitoring
                3. **Quality Planning**: Develop quality plans, control plans, and quality gates throughout product lifecycle
                4. **Risk Management**: Implement quality risk assessment methodologies (FMEA, FTA, Risk Priority Numbers)
                5. **Continuous Improvement**: Lead quality improvement initiatives using Lean Six Sigma, DMAIC, and Kaizen methodologies
                6. **Supplier Quality**: Establish supplier quality requirements, audits, and performance management
                7. **Regulatory Compliance**: Ensure adherence to industry standards and regulatory requirements
                8. **Quality Analytics**: Develop quality dashboards, trend analysis, and predictive quality models

                **Quality Framework Application:**
                - Apply Total Quality Management (TQM) principles
                - Implement Plan-Do-Check-Act (PDCA) cycles for continuous improvement
                - Utilize Statistical Quality Control (SQC) and Design of Experiments (DOE)
                - Apply quality cost models (Prevention, Appraisal, Internal/External Failure costs)
                - Implement configuration management and change control processes
                - Ensure traceability and documentation control throughout lifecycle
                - Apply quality gate criteria and stage-gate reviews

                **Technical Approach:**
                - Establish quality objectives with measurable KPIs and targets
                - Design quality control checkpoints and inspection strategies
                - Implement corrective and preventive action (CAPA) processes
                - Develop quality training and competency management programs
                - Create quality documentation hierarchies and control systems
                - Establish customer feedback loops and satisfaction measurement
                - Design quality audit programs and management review processes
                - Implement quality culture transformation and organizational change management""",
                    
                    "user_prompt": """As a senior quality engineering specialist, analyze the following quality-related artifact and provide a comprehensive assessment:

                {data_sample}

                Please provide a detailed quality engineering analysis:

                1. **Quality System Assessment** (QMS framework evaluation and compliance review)
                2. **Process Quality Analysis** (process capability, control measures, and statistical analysis)
                3. **Quality Planning Evaluation** (quality objectives, control plans, and gate criteria)
                4. **Risk Assessment Review** (quality risks identification, FMEA analysis, and mitigation strategies)
                5. **Metrics & KPI Analysis** (quality measurement systems and performance indicators)
                6. **Continuous Improvement Opportunities** (improvement initiatives and optimization recommendations)
                7. **Compliance & Standards Review** (regulatory requirements and industry standard adherence)
                8. **Cost of Quality Analysis** (prevention, appraisal, and failure cost assessment)
                9. **Supplier Quality Considerations** (supply chain quality requirements and management)
                10. **Quality Culture & Training** (organizational quality maturity and competency gaps)
                11. **Documentation & Traceability** (quality record management and audit trail adequacy)
                12. **Action Plan Development** (prioritized quality improvement roadmap and implementation strategy)

                Focus on providing systematic, data-driven recommendations that enhance quality performance, reduce defects, improve customer satisfaction, and drive organizational quality maturity.""",
                    
                    "temperature": 0.2,
                    "max_tokens": 2400
            }
        }

        # Enhanced layout with better organization (keeping your existing layout)
        col1, col2 = st.columns([1, 1])

        with col1:
            st.subheader("Basic Configuration")
            
            # Agent name with validation
            agent_name = st.text_input(
                "Agent Name", 
                placeholder="e.g., Systems Engineer Agent, Systems Test Agent, QC Agent",
                help="Choose a descriptive name that reflects the agent's specialty"
            )
            
            # Real-time validation feedback
            if agent_name:
                if len(agent_name) < 3:
                    st.warning("Agent name should be at least 3 characters")
                elif len(agent_name) > 100:
                    st.warning("Agent name should be less than 100 characters")
                else:
                    st.success("Agent name looks good")
            
            # Enhanced agent type selection with descriptions
            agent_type = st.selectbox(
                "Agent Type Template", 
                ["Custom"] + list(enhanced_templates.keys()),
                help="Select a predefined template optimized for specific tasks"
            )
            
            # Display template description
            if agent_type != "Custom":
                template = enhanced_templates[agent_type]
                st.info(f"**{agent_type}**: {template['description']}")

            # Enhanced model selection with status indicators
            st.subheader("Model Selection")
            
            # Get available models with status
            available_models = st.session_state.available_models or get_available_models_cached()
            
            if available_models:
                # Display model options with enhanced info
                model_options = []
                model_status = {}
                
                for model in available_models:
                    status = check_model_status(model)
                    if status == "available":
                        model_options.append(f"{model}")
                        model_status[model] = "available"
                    else:
                        model_options.append(f"{model}")
                        model_status[model] = "degraded"
                
                selected_model_display = st.selectbox("Select Model", model_options)
                split_parts = selected_model_display.split(" ", 1)
                agent_model_name = split_parts[1] if len(split_parts) > 1 else split_parts[0]
                
                # Display model information
                if agent_model_name in model_descriptions:
                    st.info(model_descriptions[agent_model_name])
                
                # Model status info
                status = model_status.get(agent_model_name, "unknown")
                if status == "available":
                    st.success("Model is loaded and ready")
                else:
                    st.error("Model is not available")
                    
            else:
                st.error("No models available. Please check your model configuration.")
                agent_model_name = None

        with col2:
            st.subheader("Agent Prompts & Configuration")
            
            # Auto-populate from template
            if agent_type != "Custom":
                template = enhanced_templates[agent_type]
                default_system_prompt = template["system_prompt"]
                default_user_prompt = template["user_prompt"]
                default_temperature = template.get("temperature", 0.7)
                default_max_tokens = template.get("max_tokens", 1000)
            else:
                default_system_prompt = ""
                default_user_prompt = ""
                default_temperature = 0.7
                default_max_tokens = 1000

            # System prompt with enhanced validation
            system_prompt = st.text_area(
                "System Prompt (Define Agent's Role & Expertise)",
                value=default_system_prompt,
                height=250,
                help="Define the agent's role, expertise, analysis approach, and output format",
                placeholder="You are an expert specializing in..."
            )
            
            # Character count and validation for system prompt
            if system_prompt:
                char_count = len(system_prompt)
                if char_count < 100:
                    st.warning(f"System prompt is quite short ({char_count} chars). Consider adding more detail.")
                elif char_count > 3000:
                    st.warning(f"System prompt is very long ({char_count} chars). Consider condensing.")
                else:
                    st.success(f"System prompt length is good ({char_count} chars)")

            # User prompt template with validation
            user_prompt_template = st.text_area(
                "User Prompt Template",
                value=default_user_prompt,
                height=150,
                help="Template for user queries. Must include {data_sample} placeholder for input content",
                placeholder="Analyze the following content:\n\n{data_sample}\n\nProvide detailed analysis..."
            )
            
            # Validation for user prompt template
            if user_prompt_template:
                if "{data_sample}" not in user_prompt_template:
                    st.error("User prompt template must include {data_sample} placeholder")
                else:
                    st.success("User prompt template is properly formatted")

            # Advanced configuration in expandable section
            with st.expander("Advanced Settings"):
                col1_adv, col2_adv = st.columns(2)
                
                with col1_adv:
                    temperature = st.slider(
                        "Temperature",
                        min_value=0.0,
                        max_value=1.0,
                        value=default_temperature,
                        step=0.1,
                        help="Controls creativity vs consistency. Lower = more focused and consistent responses"
                    )
                    
                    if temperature <= 0.3:
                        st.info("Low temperature: Highly consistent, focused responses")
                    elif temperature <= 0.7:
                        st.info("Medium temperature: Balanced creativity and consistency")
                    else:
                        st.info("High temperature: More creative and varied responses")
                
                with col2_adv:
                    max_tokens = st.number_input(
                        "Max Tokens",
                        min_value=100,
                        max_value=4000,
                        value=default_max_tokens,
                        step=100,
                        help="Maximum length of agent responses (roughly 1 token = 0.75 words)"
                    )
                    
                    estimated_words = int(max_tokens * 0.75)
                    st.caption(f"≈ {estimated_words} words maximum")

        # Enhanced validation and testing section (keeping your existing validation)
        # st.markdown("---")
        st.subheader("Validation & Testing")
        
        # Comprehensive validation
        validation_checks = {
            "Agent name provided": bool(agent_name and len(agent_name.strip()) >= 3),
            "Model selected": bool(agent_model_name),
            "System prompt substantial": bool(system_prompt and len(system_prompt.strip()) >= 100),
            "User prompt has {data_sample}": "{data_sample}" in user_prompt_template if user_prompt_template else False,
            "User prompt substantial": bool(user_prompt_template and len(user_prompt_template.strip()) >= 50),
            "No duplicate agent name": True  # You could add API check here
        }
        
        # Display validation results in columns
        col1_val, col2_val = st.columns(2)
        
        with col1_val:
            for check, status in list(validation_checks.items())[:3]:
                if status:
                    st.success(f"{check}")
                else:
                    st.error(f"{check}")
        
        with col2_val:
            for check, status in list(validation_checks.items())[3:]:
                if status:
                    st.success(f"{check}")
                else:
                    st.error(f"{check}")

        # Action buttons (keeping your existing create logic)
        col1_action, col2_action, col3_action = st.columns([1, 1, 1])
        
        
        with col2_action:
            if st.button("Create Agent", type="primary", use_container_width=True):
                if not all(validation_checks.values()):
                    st.warning("Please fix all validation errors before creating the agent")
                else:
                    # Enhanced agent payload
                    agent_payload = {
                        "name": agent_name.strip(),
                        "model_name": agent_model_name,
                        "system_prompt": system_prompt.strip(),
                        "user_prompt_template": user_prompt_template.strip()
                    }
                    
                    with st.spinner(f"Creating agent '{agent_name}'..."):
                        try:
                            response = requests.post(
                                f"{FASTAPI_URL}/create-agent", 
                                json=agent_payload, 
                                timeout=30
                            )
                            
                            if response.status_code == 200:
                                result = response.json()
                                st.success(f"Agent '{agent_name}' created successfully!")
                                
                                # Display creation details
                                st.info(f"Agent ID: {result.get('agent_id', 'Unknown')}")
                                st.info(f"Model: {agent_model_name}")
                                st.info(f"Type: {agent_type}")
                                
                                # Clear session state to force refresh
                                st.session_state.agents_data = []
                                
                                # Quick action suggestions
                                st.markdown("**Next Steps:**")
                                st.markdown("- Switch to '🤖 AI Agent Simulation' mode to test your new agent")
                                st.markdown("- Use 'Manage Existing Agents' tab to edit or manage agents")
                                st.markdown("- Create additional specialized agents for different tasks")
                                
                            else:
                                error_detail = response.json().get("detail", response.text) if response.headers.get("content-type") == "application/json" else response.text
                                
                                if "already exists" in error_detail.lower():
                                    st.error(f"Agent name '{agent_name}' already exists. Please choose a different name.")
                                else:
                                    st.error(f"Failed to create agent: {error_detail}")
                                    
                        except requests.exceptions.Timeout:
                            st.error("Request timed out. Please try again.")
                        except requests.exceptions.ConnectionError:
                            st.error("Cannot connect to the API. Please check if the server is running.")
                        except Exception as e:
                            st.error(f"Unexpected error: {str(e)}")

    # ----------------------------------------------------------------------
    # MANAGE EXISTING AGENTS SUB-MODE
    # ----------------------------------------------------------------------
    elif agent_mode == "Manage Existing Agents":
        st.subheader("Manage Existing Legal Agents")
        
        # Load agents with enhanced error handling
        col1_load, col2_load = st.columns([1, 2])
        
        with col1_load:
            if st.button("Refresh Agent List", key="manage_refresh"):
                try:
                    with st.spinner("Loading agents from database..."):
                        agents_response = requests.get(f"{FASTAPI_URL}/get-agents", timeout=10)
                        if agents_response.status_code == 200:
                            agent_data = agents_response.json()
                            st.session_state.agents_data = agent_data.get("agents", [])
                            st.success(f"Loaded {len(st.session_state.agents_data)} agents")
                        else:
                            st.warning(f"Could not load agents (Status: {agents_response.status_code})")
                except Exception as e:
                    st.error(f"Error loading agents: {e}")
        
        with col2_load:
            if st.session_state.agents_data:
                total_agents = len(st.session_state.agents_data)
                active_agents = sum(1 for agent in st.session_state.agents_data if agent.get("is_active", True))
                st.metric("Total Agents", total_agents, delta=f"{active_agents} active")

        # Enhanced agents display
        if st.session_state.agents_data:
            # Create enhanced table data
            agents_data = []
            for agent in st.session_state.agents_data:
                agents_data.append({
                    "ID": agent.get("id", "N/A"),
                    "Name": agent.get("name", "Unknown"),
                    "Model": agent.get("model_name", "Unknown"),
                    "Queries": agent.get("total_queries", 0),
                    "Status": "Active" if agent.get("is_active", True) else "Inactive",
                    "Created": agent.get("created_at", "Unknown")[:10] if agent.get("created_at") else "Unknown",
                    "System Prompt": agent.get("system_prompt", "")[:100] + ("..." if len(agent.get("system_prompt", "")) > 100 else ""),
                    "User Template": agent.get("user_prompt_template", "")[:100] + ("..." if len(agent.get("user_prompt_template", "")) > 100 else "")
                })
            
            # Display in a nice table
            st.dataframe(agents_data, use_container_width=True, height=400)
            
            # Management actions
            management_action = st.radio(
                "Management Action:",
                ["📋 View Details", "✏️ Edit Agent", "🗑️ Delete Agent"],
                horizontal=True,
                key="mgmt_action"
            )
            
            # Agent selection for all actions
            agent_options = [f"{agent['name']} (ID: {agent['id']})" for agent in st.session_state.agents_data]
            selected_agent_str = st.selectbox(
                "Select Agent:",
                ["--Select Agent--"] + agent_options,
                key="selected_agent_mgmt"
            )
            
            if selected_agent_str != "--Select Agent--":
                agent_id = int(selected_agent_str.split("ID: ")[1].rstrip(")"))
                selected_agent = next((agent for agent in st.session_state.agents_data if agent["id"] == agent_id), None)
                
                if selected_agent:
                    # VIEW DETAILS
                    if management_action == "View Details":
                        st.subheader(f"Agent Details: {selected_agent['name']}")
                        
                        col1_details, col2_details = st.columns(2)
                        
                        with col1_details:
                            st.markdown("**Basic Information:**")
                            st.json({
                                "ID": selected_agent.get("id"),
                                "Name": selected_agent.get("name"),
                                "Model": selected_agent.get("model_name"),
                                "Created": selected_agent.get("created_at"),
                                "Updated": selected_agent.get("updated_at"),
                                "Status": "Active" if selected_agent.get("is_active", True) else "Inactive"
                            })
                            
                            st.markdown("**Performance Metrics:**")
                            st.json({
                                "Total Queries": selected_agent.get("total_queries", 0),
                                "Temperature": selected_agent.get("temperature", 0.7),
                                "Max Tokens": selected_agent.get("max_tokens", 300)
                            })
                        
                        with col2_details:
                            st.markdown("**System Prompt:**")
                            st.text_area(
                                "Full System Prompt", 
                                selected_agent.get("system_prompt", ""), 
                                height=200, 
                                disabled=True,
                                key="view_system_prompt"
                            )
                            
                            st.markdown("**User Prompt Template:**")
                            st.text_area(
                                "Full User Prompt Template", 
                                selected_agent.get("user_prompt_template", ""), 
                                height=150, 
                                disabled=True,
                                key="view_user_prompt"
                            )
                    
                    # EDIT AGENT
                    elif management_action == "✏️ Edit Agent":
                        st.subheader(f"✏️ Edit Agent: {selected_agent['name']}")
                        
                        with st.form(f"edit_agent_{agent_id}"):
                            col1_edit, col2_edit = st.columns(2)
                            
                            with col1_edit:
                                new_name = st.text_input(
                                    "Agent Name", 
                                    value=selected_agent.get("name", ""),
                                    key="edit_name"
                                )
                                
                                # Model selection for edit
                                available_models = st.session_state.available_models or get_available_models_cached()
                                current_model = selected_agent.get("model_name", "")
                                
                                if current_model in available_models:
                                    model_index = available_models.index(current_model)
                                else:
                                    model_index = 0
                                
                                new_model = st.selectbox(
                                    "Model", 
                                    available_models, 
                                    index=model_index,
                                    key="edit_model"
                                )
                                
                                # Advanced settings
                                new_temperature = st.slider(
                                    "Temperature",
                                    min_value=0.0,
                                    max_value=1.0,
                                    value=selected_agent.get("temperature", 0.7),
                                    step=0.1,
                                    key="edit_temperature"
                                )
                                
                                new_max_tokens = st.number_input(
                                    "Max Tokens",
                                    min_value=100,
                                    max_value=4000,
                                    value=selected_agent.get("max_tokens", 300),
                                    step=100,
                                    key="edit_max_tokens"
                                )
                                
                                new_is_active = st.checkbox(
                                    "Active", 
                                    value=selected_agent.get("is_active", True),
                                    key="edit_active"
                                )
                            
                            with col2_edit:
                                new_system_prompt = st.text_area(
                                    "System Prompt",
                                    value=selected_agent.get("system_prompt", ""),
                                    height=200,
                                    key="edit_system_prompt"
                                )
                                
                                new_user_prompt = st.text_area(
                                    "User Prompt Template",
                                    value=selected_agent.get("user_prompt_template", ""),
                                    height=150,
                                    key="edit_user_prompt"
                                )
                            
                            # Form submission
                            submitted = st.form_submit_button("💾 Update Agent", type="primary")
                            
                            if submitted:
                                # Validation
                                if not new_name or len(new_name.strip()) < 3:
                                    st.error("Agent name must be at least 3 characters")
                                elif "{data_sample}" not in new_user_prompt:
                                    st.error("User prompt template must contain {data_sample} placeholder")
                                else:
                                    # Prepare update payload
                                    update_payload = {
                                        "name": new_name.strip(),
                                        "model_name": new_model,
                                        "system_prompt": new_system_prompt.strip(),
                                        "user_prompt_template": new_user_prompt.strip(),
                                        "temperature": new_temperature,
                                        "max_tokens": new_max_tokens,
                                        "is_active": new_is_active
                                    }
                                    
                                    try:
                                        with st.spinner("Updating agent..."):
                                            response = requests.put(
                                                f"{FASTAPI_URL}/update-agent/{agent_id}",
                                                json=update_payload,
                                                timeout=30
                                            )
                                            
                                            if response.status_code == 200:
                                                st.success(f"Agent '{new_name}' updated successfully!")
                                                st.session_state.agents_data = []  # Force refresh
                                                st.rerun()
                                            else:
                                                error_detail = response.json().get("detail", response.text) if response.headers.get("content-type") == "application/json" else response.text
                                                st.error(f"Failed to update agent: {error_detail}")
                                    
                                    except Exception as e:
                                        st.error(f"Error updating agent: {str(e)}")
                    
                    # DELETE AGENT
                    elif management_action == "🗑️ Delete Agent":
                        st.subheader(f"🗑️ Delete Agent: {selected_agent['name']}")
                        
                        st.warning("**Permanent Action**: Agent deletion cannot be undone and will remove all associated data.")
                        
                        # Show agent info before deletion
                        with st.expander("Agent to be deleted", expanded=True):
                            st.json({
                                "ID": selected_agent.get("id"),
                                "Name": selected_agent.get("name"),
                                "Model": selected_agent.get("model_name"),
                                "Total Queries": selected_agent.get("total_queries", 0),
                                "Created": selected_agent.get("created_at")
                            })
                        
                        # Confirmation steps
                        confirm_name = st.text_input(
                            f"Type the agent name '{selected_agent['name']}' to confirm deletion:",
                            key="delete_confirm_name"
                        )
                        
                        confirm_delete = st.checkbox(
                            f"I understand this will permanently delete agent: {selected_agent['name']}",
                            key="delete_confirm_checkbox"
                        )
                        
                        if confirm_name == selected_agent['name'] and confirm_delete:
                            if st.button("🗑️ Confirm Deletion", type="secondary", key="delete_confirm_button"):
                                try:
                                    with st.spinner("Deleting agent..."):
                                        response = requests.delete(f"{FASTAPI_URL}/delete-agent/{agent_id}", timeout=10)
                                        
                                        if response.status_code == 200:
                                            st.success("Agent deleted successfully!")
                                            # Force refresh of agents list
                                            st.session_state.agents_data = []
                                            st.rerun()
                                        else:
                                            st.error(f"Failed to delete agent: HTTP {response.status_code}")
                                            
                                except Exception as e:
                                    st.error(f"Error deleting agent: {e}")
                        else:
                            if confirm_name != selected_agent['name'] and confirm_name:
                                st.error("Agent name doesn't match")
                            if not confirm_delete:
                                st.info("Please check the confirmation box and enter the exact agent name to enable deletion")

        else:
            # Enhanced empty state
            st.info("No agents found. Create your first legal agent in the 'Create New Agent' tab!")
            
            # Quick start guide
            with st.expander("Quick Start Guide"):
                st.markdown("""
                **Getting Started with Agents:**
                
                1. **Switch to 'Create New Agent'** tab above
                2. **Choose a Template**: Select from predefined templates like 'Systems Engineer' or 'Quality Control Engineer'
                3. **Configure Settings**: Adjust temperature (creativity) and max tokens (response length)
                4. **Test Configuration**: Use the test button to preview how your agent will work
                5. **Create Agent**: Click 'Create Agent' to add it to your legal AI toolkit
                6. **Come back here to manage**: Edit, view details, or delete agents
                """)

    # Enhanced tips and best practices (shown for both modes)
    st.markdown("---")
    with st.expander("Agent Best Practices & Tips", expanded=False):
        col1_tips, col2_tips = st.columns(2)
        
        with col1_tips:
            st.markdown("""
            **System Prompt Best Practices:**
            - Define specific expertise areas clearly
            - Include analysis frameworks and methodologies
            - Specify output format and structure
            - Add relevant legal standards or regulations
            - Use bullet points for clarity
            - Include examples of what to focus on
            """)
            
            st.markdown("""
            **Management Tips:**
            - Regularly review agent performance metrics
            - Update prompts based on usage patterns
            - Deactivate unused agents to keep interface clean
            - Test agents after making changes
            """)
            
        with col2_tips:
            st.markdown("""
            **User Prompt Template Tips:**
            - Always include `{data_sample}` placeholder
            - Provide clear instructions for analysis type
            - Specify desired output structure
            - Include relevant context or constraints
            - Ask for specific recommendations
            - Consider different input types (contracts, policies, etc.)
            """)
            
            st.markdown("""
            **Performance Optimization:**
            - Lower temperature (0.1-0.3) for consistent compliance checks
            - Higher temperature (0.7-0.9) for creative legal brainstorming
            - Adjust max tokens based on typical response needs
            - Monitor success rates and response times
            """)
        
        st.markdown("""
        **Model Selection Guidelines:**
        - **LLaMA**: Fast local processing, good for privacy-sensitive content
        
        **Temperature Settings:**
        - **0.1-0.3**: Highly consistent, factual analysis (recommended for compliance)
        - **0.4-0.7**: Balanced creativity and consistency (good for general legal work)
        - **0.8-1.0**: More creative responses (useful for brainstorming or alternative approaches)
        """)

    # Footer for create agent section
    st.markdown("---")
    st.warning("**Agent Disclaimer**: All created agents provide analysis for informational purposes only and do not constitute advice.")
    st.info("🔒 **Data Security**: Ensure all content processed by agents complies with your organization's data protection and confidentiality policies.")
    
    

# ----------------------------------------------------------------------
# SESSION HISTORY & ANALYTICS MODE
# ----------------------------------------------------------------------
elif chat_mode == "📊 Session History":
    st.markdown("---")
    st.header("📊 Agent Session History & Analytics")
    
    # Sub-mode selection
    history_mode = st.radio(
        "View:",
        ["📜 Recent Sessions", "📈 Analytics Dashboard", "🔍 Session Details"],
        horizontal=True
    )
    
    if history_mode == "📜 Recent Sessions":
        st.subheader("Recent Agent Sessions")
        
        # Filters
        col1, col2, col3 = st.columns(3)
        
        with col1:
            session_limit = st.number_input("Number of sessions", min_value=10, max_value=200, value=50, step=10)
        
        with col2:
            session_type_filter = st.selectbox(
                "Filter by type:",
                ["All", "single_agent", "multi_agent_debate", "rag_analysis", "rag_debate", "compliance_check"]
            )
        
        with col3:
            if st.button("Load Sessions"):
                try:
                    # Prepare API call
                    params = {"limit": session_limit}
                    if session_type_filter != "All":
                        params["session_type"] = session_type_filter
                    
                    with st.spinner("Loading session history..."):
                        response = requests.get(f"{FASTAPI_URL}/session-history", params=params, timeout=10)
                        
                        if response.status_code == 200:
                            data = response.json()
                            st.session_state.session_history = data.get("sessions", [])
                            st.success(f"Loaded {len(st.session_state.session_history)} sessions")
                        else:
                            st.error(f"Failed to load sessions: {response.status_code}")
                            
                except Exception as e:
                    st.error(f"Error: {e}")
        
        # Display sessions
        if 'session_history' in st.session_state and st.session_state.session_history:
            sessions = st.session_state.session_history
            
            # Convert to DataFrame for better display
            session_data = []
            for session in sessions:
                session_data.append({
                    "Session ID": session["session_id"][:8] + "...",
                    "Type": session["session_type"].replace("_", " ").title(),
                    "Analysis": session["analysis_type"].replace("_", " ").title(),
                    "Query Preview": session["user_query"][:100] + "..." if len(session["user_query"]) > 100 else session["user_query"],
                    "Collection": session.get("collection_name", "N/A"),
                    "Agents": session.get("agent_count", 0),
                    "Response Time": f"{session.get('total_response_time_ms', 0)}ms" if session.get('total_response_time_ms') else "N/A",
                    "Status": session["status"].title(),
                    "Created": session["created_at"][:19] if session["created_at"] else "Unknown"
                })
            
            st.dataframe(session_data, use_container_width=True, height=400)
            
            # Session details viewer
            st.subheader("🔍 View Session Details")
            session_ids = [s["session_id"] for s in sessions]
            selected_session = st.selectbox(
                "Select session to view details:",
                ["--Select Session--"] + session_ids
            )
            
            if selected_session != "--Select Session--":
                if st.button("📋 Load Session Details"):
                    try:
                        with st.spinner("Loading session details..."):
                            response = requests.get(f"{FASTAPI_URL}/session-details/{selected_session}", timeout=10)
                            
                            if response.status_code == 200:
                                details = response.json()
                                st.session_state.session_details = details
                                
                                # Display session info
                                session_info = details["session_info"]
                                
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.json({
                                        "Session ID": session_info["session_id"],
                                        "Type": session_info["session_type"],
                                        "Analysis Type": session_info["analysis_type"],
                                        "Status": session_info["status"],
                                        "Agent Count": session_info["agent_count"],
                                        "Total Time": f"{session_info.get('total_response_time_ms', 0)}ms"
                                    })
                                
                                with col2:
                                    st.text_area("User Query", session_info["user_query"], height=150, disabled=True)
                                
                                # Display agent responses
                                st.subheader("🤖 Agent Responses")
                                responses = details["agent_responses"]
                                
                                for i, response in enumerate(responses, 1):
                                    sequence = response.get("sequence_order", i)
                                    agent_name = response["agent_name"]
                                    
                                    with st.expander(f"Response {sequence}: {agent_name}", expanded=(i <= 2)):
                                        col1_resp, col2_resp = st.columns([2, 1])
                                        
                                        with col1_resp:
                                            st.text_area(
                                                "Response", 
                                                response["response_text"], 
                                                height=200, 
                                                disabled=True,
                                                key=f"response_{i}"
                                            )
                                        
                                        with col2_resp:
                                            st.json({
                                                "Agent ID": response["agent_id"],
                                                "Model": response["model_used"],
                                                "Method": response["processing_method"],
                                                "Time": f"{response.get('response_time_ms', 0)}ms",
                                                "RAG Used": response.get("rag_used", False),
                                                "Docs Found": response.get("documents_found", 0)
                                            })
                            else:
                                st.error(f"Failed to load details: {response.status_code}")
                                
                    except Exception as e:
                        st.error(f"Error: {e}")
        
        else:
            st.info("Click 'Load Sessions' to view recent agent sessions")
    
    elif history_mode == "📈 Analytics Dashboard":
        st.subheader("Analytics Dashboard")
        
        # Time period selection
        col1, col2 = st.columns([1, 3])
        
        with col1:
            days = st.selectbox("Time Period:", [1, 7, 14, 30], index=1)
        
        with col2:
            if st.button("📊 Load Analytics"):
                try:
                    with st.spinner("Loading analytics..."):
                        response = requests.get(f"{FASTAPI_URL}/session-analytics?days={days}", timeout=10)
                        
                        if response.status_code == 200:
                            analytics = response.json()
                            st.session_state.analytics = analytics
                            st.success(f"Loaded analytics for last {days} days")
                        else:
                            st.error(f"Failed to load analytics: {response.status_code}")
                            
                except Exception as e:
                    st.error(f"Error: {e}")
        
        # Display analytics
        if 'analytics' in st.session_state:
            analytics = st.session_state.analytics
            
            # Summary metrics
            col1, col2, col3, col4 = st.columns(4)
            
            session_stats = analytics["session_statistics"]
            rag_stats = analytics["rag_statistics"]
            
            with col1:
                total_sessions = sum(session_stats["by_session_type"].values())
                st.metric("Total Sessions", total_sessions)
            
            with col2:
                avg_time = session_stats["avg_response_time_ms"]
                st.metric("Avg Response Time", f"{avg_time:.0f}ms")
            
            with col3:
                st.metric("Total Responses", rag_stats["total_responses"])
            
            with col4:
                rag_rate = rag_stats["rag_usage_rate"]
                st.metric("RAG Usage Rate", f"{rag_rate:.1f}%")
            
            # Charts
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Sessions by Type")
                session_types = session_stats["by_session_type"]
                if session_types:
                    st.bar_chart(session_types)
                else:
                    st.info("No session data available")
            
            with col2:
                st.subheader("Analysis Types")
                analysis_types = session_stats["by_analysis_type"]
                if analysis_types:
                    st.bar_chart(analysis_types)
                else:
                    st.info("No analysis data available")
            
            # Agent activity
            st.subheader("Most Active Agents")
            agent_activity = analytics["agent_activity"]
            
            if agent_activity:
                agent_data = {agent["agent_name"]: agent["response_count"] for agent in agent_activity}
                st.bar_chart(agent_data)
                
                # Detailed table
                agent_table = [
                    {
                        "Agent Name": agent["agent_name"],
                        "Agent ID": agent["agent_id"],
                        "Response Count": agent["response_count"]
                    }
                    for agent in agent_activity
                ]
                st.dataframe(agent_table, use_container_width=True)
            else:
                st.info("No agent activity data available")
        
        else:
            st.info("👆 Click 'Load Analytics' to view performance metrics")
    
    elif history_mode == "🔍 Session Details":
        st.subheader("Search Session Details")
        
        # Manual session ID input
        session_id_input = st.text_input("Enter Session ID:")
        
        if st.button("🔍 Search Session") and session_id_input:
            try:
                with st.spinner("Searching for session..."):
                    response = requests.get(f"{FASTAPI_URL}/session-details/{session_id_input}", timeout=10)
                    
                    if response.status_code == 200:
                        details = response.json()
                        
                        # Display detailed session information
                        session_info = details["session_info"]
                        
                        st.success(f"Found session: {session_info['session_type']}")
                        
                        # Session metadata
                        with st.expander("📋 Session Information", expanded=True):
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.json({
                                    "Session ID": session_info["session_id"],
                                    "Type": session_info["session_type"],
                                    "Analysis Type": session_info["analysis_type"],
                                    "Created": session_info["created_at"],
                                    "Completed": session_info["completed_at"],
                                    "Status": session_info["status"]
                                })
                            
                            with col2:
                                st.json({
                                    "Agent Count": session_info["agent_count"],
                                    "Total Time": f"{session_info.get('total_response_time_ms', 0)}ms",
                                    "Collection": session_info.get("collection_name", "N/A"),
                                    "Error": session_info.get("error_message", "None")
                                })
                        
                        # User query
                        st.text_area("User Query", session_info["user_query"], height=100, disabled=True)
                        
                        # Agent responses
                        responses = details["agent_responses"]
                        st.subheader(f"🤖 Agent Responses ({len(responses)})")
                        
                        for response in responses:
                            agent_name = response["agent_name"]
                            sequence = response.get("sequence_order", "N/A")
                            
                            with st.expander(f"Agent: {agent_name} (Sequence: {sequence})", expanded=False):
                                st.text_area("Response", response["response_text"], height=200, disabled=True)
                                
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.json({
                                        "Processing Method": response["processing_method"],
                                        "Response Time": f"{response.get('response_time_ms', 0)}ms",
                                        "Model Used": response["model_used"]
                                    })
                                
                                with col2:
                                    st.json({
                                        "RAG Used": response.get("rag_used", False),
                                        "Documents Found": response.get("documents_found", 0),
                                        "Compliant": response.get("compliant", "N/A"),
                                        "Confidence": response.get("confidence_score", "N/A")
                                    })
                    
                    elif response.status_code == 404:
                        st.warning("Session not found")
                    else:
                        st.error(f"Error: {response.status_code}")
                        
            except Exception as e:
                st.error(f"Error: {e}")


        

# Footer
st.markdown("---")
st.caption("🔒 This application processes documents and provide GenAI capabilitites. Ensure all data is handled according to your organization's data protection policies.")
