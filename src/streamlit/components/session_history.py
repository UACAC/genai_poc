import streamlit as st
import requests
from utils import * 

FASTAPI_API = os.getenv("FASTAPI_URL", "http://localhost:9020")

def Session_History():
    st.header("Agent Session History & Analytics")
    
    # Sub-mode selection
    history_mode = st.radio(
        "View:",
        ["Recent Sessions", "Analytics Dashboard", "Session Details"],
        horizontal=True
    )
    
    if history_mode == "Recent Sessions":
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
                        response = requests.get(f"{FASTAPI_API}/session-history", params=params, timeout=10)
                        
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
            st.subheader("View Session Details")
            session_ids = [s["session_id"] for s in sessions]
            selected_session = st.selectbox(
                "Select session to view details:",
                ["--Select Session--"] + session_ids
            )
            
            if selected_session != "--Select Session--":
                if st.button("Load Session Details"):
                    try:
                        with st.spinner("Loading session details..."):
                            response = requests.get(f"{FASTAPI_API}/session-details/{selected_session}", timeout=10)
                            
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
                                st.subheader("Agent Responses")
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
    
    elif history_mode == "Analytics Dashboard":
        st.subheader("Analytics Dashboard")
        
        # Time period selection
        col1, col2 = st.columns([1, 3])
        
        with col1:
            days = st.selectbox("Time Period:", [1, 7, 14, 30], index=1)
        
        with col2:
            if st.button("Load Analytics"):
                try:
                    with st.spinner("Loading analytics..."):
                        response = requests.get(f"{FASTAPI_API}/session-analytics?days={days}", timeout=10)
                        
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
            st.info("Click 'Load Analytics' to view performance metrics")
    
    elif history_mode == "Session Details":
        st.subheader("Search Session Details")
        
        # Manual session ID input
        session_id_input = st.text_input("Enter Session ID:")
        
        if st.button("Search Session") and session_id_input:
            try:
                with st.spinner("Searching for session..."):
                    response = requests.get(f"{FASTAPI_API}/session-details/{session_id_input}", timeout=10)
                    
                    if response.status_code == 200:
                        details = response.json()
                        
                        # Display detailed session information
                        session_info = details["session_info"]
                        
                        st.success(f"Found session: {session_info['session_type']}")
                        
                        # Session metadata
                        with st.expander("Session Information", expanded=True):
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
                        st.subheader(f"Agent Responses ({len(responses)})")
                        
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