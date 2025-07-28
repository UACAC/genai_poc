import streamlit as st
import requests
from utils import * 


def Agent_Sim():
    FASTAPI_API = os.getenv("FASTAPI_URL", "http://localhost:9020")
    # Display collections
    collections = st.session_state.collections
    if collections:
        for collection in collections:
            st.text(f"{collection}")
    else:
        st.info("Click 'Load Collections' to see available databases")
    
    # Load agents with enhanced error handling
    col1_load, col2_load = st.columns([1, 2])
    
    with col1_load:
        if st.button("Refresh Agent List"):
            try:
                with st.spinner("Loading agents from database..."):
                    agents_response = requests.get(f"{FASTAPI_API}/get-agents", timeout=10)
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
            "Content for Agent Analysis", 
            placeholder="Paste contract text,documents, or content for analysis...",
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
                    endpoint = f"{FASTAPI_API}/rag-check"
                else:
                    payload = {
                        "data_sample": analysis_content,
                        "agent_ids": agent_ids
                    }
                    endpoint = f"{FASTAPI_API}/compliance-check"
                
                with st.spinner("Specialized agents are analyzing the content..."):
                    status_placeholder = st.empty()
                    try:
                        status_placeholder.info("Connecting to AI model...")
                        response = requests.post(endpoint, json=payload, timeout=300)
                        
                        if response.status_code == 200:
                            result = response.json()
                            
                            # Display results
                            agent_responses = result.get("agent_responses", {})
                            if agent_responses:
                                for agent_name, analysis in agent_responses.items():
                                    with st.expander(f"{agent_name} Analysis", expanded=True):
                                        st.markdown(analysis)
                            else:
                                # Handle compliance check format
                                details = result.get("details", {})
                                for idx, analysis in details.items():
                                    agent_name = analysis.get("agent_name", f"Agent {idx}")
                                    reason = analysis.get("reason", analysis.get("raw_text", "No analysis"))
                                    
                                    with st.expander(f"{agent_name} Analysis", expanded=True):
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
        st.subheader("Multi-Agent Debate Sequence")
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
            if st.button("+ Add to Sequence", key="add_agent_debate"):
                if new_agent_to_add != "--Select an Agent--" and new_agent_to_add not in st.session_state["debate_sequence"]:
                    st.session_state["debate_sequence"].append(new_agent_to_add)
                    st.success(f"Added {new_agent_to_add}")
                    st.rerun()
                elif new_agent_to_add in st.session_state["debate_sequence"]:
                    st.warning("Agent already in sequence!")
            
            if st.button("Clear All Sequences"):
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
                    if st.button("Remove", key=f"remove_{i}", help="Remove from sequence"):
                        st.session_state["debate_sequence"].remove(agent_name)
                        st.rerun()
            
            st.markdown("---")
        else:
            st.info("Add agents to create a debate sequence")
        
        # Content and collection selection for debate
        debate_content = st.text_area(
            "Content for Multi-Agent Debate", 
            placeholder="Enter the content that agents will debate about...",
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
        if st.button("Start Multi-Agent Debate", type="primary", key="start_debate"):
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
                    with st.expander("Debate Setup", expanded=True):
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
                        endpoint = f"{FASTAPI_API}/rag-debate-sequence"
                    else:
                        debate_payload = {
                            "data_sample": debate_content,
                            "agent_ids": sequence_agent_ids
                        }
                        endpoint = f"{FASTAPI_API}/compliance-check"  # Use compliance check for non-RAG debate
                    
                    # Start the debate
                    with st.spinner(f"{len(sequence_agent_ids)} agents are debating..."):
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
                                    st.subheader("Debate Sequence Results")
                                    
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
                                    st.subheader("Agent Analysis Results")
                                    details = result["details"]
                                    
                                    for idx, analysis in details.items():
                                        agent_name = analysis.get("agent_name", f"Agent {idx}")
                                        reason = analysis.get("reason", analysis.get("raw_text", "No analysis"))
                                        
                                        with st.expander(f"{agent_name}", expanded=True):
                                            st.markdown(reason)
                                
                                elif "agent_responses" in result:
                                    # Handle agent_responses format
                                    st.subheader("Agent Analysis Results")
                                    agent_responses = result["agent_responses"]
                                    
                                    for agent_name, response_text in agent_responses.items():
                                        with st.expander(f"{agent_name}", expanded=True):
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
        st.info("Click 'Refresh Agent List' to load your specialized agents")
        st.markdown("""
        **No agents found!** 
        
        To use Agent Simulation mode:
        1. Go to the 'Create Agent' tab
        2. Create some specialized agents
        3. Come back here to simulate multi-agent analysis
        """)

