import streamlit as st
import requests
from utils import *

FASTAPI_API = os.getenv("FASTAPI_URL", "http://localhost:9020")

def AI_Agent():
    st.header("Agent Management")
    
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

                "temperature": 0.2,
                "max_tokens": 1000
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
                    "max_tokens": 1000
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
                    "max_tokens": 1000
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
            available_models = get_available_models_cached()
            
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
                default_temperature = template.get("temperature", 0.2)
                default_max_tokens = template.get("max_tokens", 1000)
            else:
                default_system_prompt = ""
                default_user_prompt = ""
                default_temperature = 0.2
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
                                f"{FASTAPI_API}/create-agent", 
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
                                st.markdown("- Switch to 'AI Agent Simulation' mode to test your new agent")
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
        st.subheader("Manage Existing Agents")
        
        # Load agents with enhanced error handling
        col1_load, col2_load = st.columns([1, 2])
        
        with col1_load:
            if st.button("Refresh Agent List", key="manage_refresh"):
                try:
                    with st.spinner("Loading agents from database..."):
                        agents_response = requests.get(f"{FASTAPI_API}/get-agents", timeout=10)
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
                ["View Details", "Edit Agent", "Delete Agent"],
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
                    elif management_action == "Edit Agent":
                        st.subheader(f"Edit Agent: {selected_agent['name']}")
                        
                        with st.form(f"edit_agent_{agent_id}"):
                            col1_edit, col2_edit = st.columns(2)
                            
                            with col1_edit:
                                new_name = st.text_input(
                                    "Agent Name", 
                                    value=selected_agent.get("name", ""),
                                    key="edit_name"
                                )
                                
                                # Model selection for edit
                                available_models = get_available_models_cached()
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
                                                f"{FASTAPI_API}/update-agent/{agent_id}",
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
                    elif management_action == "Delete Agent":
                        st.subheader(f"Delete Agent: {selected_agent['name']}")
                        
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
                            if st.button("Confirm Deletion", type="secondary", key="delete_confirm_button"):
                                try:
                                    with st.spinner("Deleting agent..."):
                                        response = requests.delete(f"{FASTAPI_API}/delete-agent/{agent_id}", timeout=10)
                                        
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
            st.info("No agents found. Create your first agent in the 'Create New Agent' tab!")
            
            # Quick start guide
            with st.expander("Quick Start Guide"):
                st.markdown("""
                **Getting Started with Agents:**
                
                1. **Switch to 'Create New Agent'** tab above
                2. **Choose a Template**: Select from predefined templates like 'Systems Engineer' or 'Quality Control Engineer'
                3. **Configure Settings**: Adjust temperature (creativity) and max tokens (response length)
                4. **Test Configuration**: Use the test button to preview how your agent will work
                5. **Create Agent**: Click 'Create Agent' to add it to your AI toolkit
                6. **Come back here to manage**: Edit, view details, or delete agents
                """)

    # Enhanced tips and best practices (shown for both modes)
    with st.expander("Agent Best Practices & Tips", expanded=False):
        col1_tips, col2_tips = st.columns(2)
        
        with col1_tips:
            st.markdown("""
            **System Prompt Best Practices:**
            - Define specific expertise areas clearly
            - Include analysis frameworks and methodologies
            - Specify output format and structure
            - Add relevant standards or regulations
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
            - Higher temperature (0.7-0.9) for creative brainstorming
            - Adjust max tokens based on typical response needs
            - Monitor success rates and response times
            """)
        
        st.markdown("""
        **Model Selection Guidelines:**
        - **LLaMA**: Fast local processing, good for privacy-sensitive content
        
        **Temperature Settings:**
        - **0.1-0.3**: Highly consistent, factual analysis (recommended for compliance)
        - **0.4-0.7**: Balanced creativity and consistency (good for general work)
        - **0.8-1.0**: More creative responses (useful for brainstorming or alternative approaches)
        """)