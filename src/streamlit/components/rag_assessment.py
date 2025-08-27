import streamlit as st
import requests
import json
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from typing import Dict, List, Any
import os
from utils import get_chromadb_collections

FASTAPI_API = os.getenv("FASTAPI_URL", "http://localhost:9020")

def RAGAS_Dashboard():
    """
    RAG Assessment Service Dashboard - Comprehensive RAG performance monitoring and analytics
    """
    st.header("RAG Assessment Service (RAG-AS)")
    st.info("Monitor, evaluate, and optimize RAG performance with comprehensive metrics and analytics.")
    
    # Health check
    try:
        health_response = requests.get(f"{FASTAPI_API}/rag-health", timeout=5)
        if health_response.status_code == 200:
            health_data = health_response.json()
            st.success(f"RAG Assessment Service is healthy - {health_data['metrics']['active_sessions']} sessions tracked")
        else:
            st.warning("RAG Assessment Service health check failed")
    except Exception as e:
        st.error(f"Cannot connect to RAG Assessment Service: {str(e)}")
        return
    
    # Main tabs
    assessment_tab, analytics_tab, benchmark_tab, collections_tab, export_tab = st.tabs([
        "Live Assessment", 
        "Performance Analytics", 
        "Benchmarking", 
        "Collection Analysis", 
        "Export & Reports"
    ])
    
    # ===== LIVE ASSESSMENT TAB =====
    with assessment_tab:
        st.subheader("Live RAG Assessment")
        st.info("Test individual queries with comprehensive performance and quality metrics.")
        
        # Load collections
        collections = get_chromadb_collections()
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            query_text = st.text_area(
                "Query for RAG Assessment:",
                placeholder="e.g., 'What are the key compliance requirements in this contract?'",
                height=100,
                key="rag_assessment_query"
            )
        
        with col2:
            if collections:
                collection_name = st.selectbox(
                    "Select Collection:",
                    collections,
                    key="rag_assessment_collection"
                )
            else:
                st.warning("No collections available")
                collection_name = None
            
            model_name = st.selectbox(
                "Model:",
                ["gpt-4","gpt-3.5-turbo"],
                key="rag_assessment_model"
            )
            
            top_k = st.slider("Top K Documents:", 1, 20, 5, key="rag_assessment_top_k")
            
            include_quality = st.checkbox("Include Quality Assessment", value=True, key="rag_quality_check")
        
        if st.button("Run RAG Assessment", type="primary", key="run_rag_assessment"):
            if not query_text:
                st.warning("Please enter a query.")
            elif not collection_name:
                st.warning("Please select a collection.")
            else:
                with st.spinner("Performing comprehensive RAG assessment..."):
                    try:
                        payload = {
                            "query": query_text,
                            "collection_name": collection_name,
                            "model_name": model_name,
                            "top_k": top_k,
                            "include_quality_assessment": include_quality
                        }
                        
                        response = requests.post(
                            f"{FASTAPI_API}/rag-assessment",
                            json=payload,
                            timeout=300
                        )
                        
                        if response.status_code == 200:
                            result = response.json()
                            
                            # Display response
                            st.success("RAG Assessment Complete!")
                            
                            # Response content
                            st.subheader("Generated Response")
                            st.markdown(result["response"])
                            
                            # Performance metrics
                            st.subheader("Performance Metrics")
                            perf = result["performance_metrics"]
                            
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("Total Time", f"{perf['total_time_ms']:.1f}ms")
                            with col2:
                                st.metric("Retrieval Time", f"{perf['retrieval_time_ms']:.1f}ms")
                            with col3:
                                st.metric("Generation Time", f"{perf['generation_time_ms']:.1f}ms")
                            with col4:
                                st.metric("Documents Retrieved", perf['documents_retrieved'])
                            
                            # Performance breakdown chart
                            perf_data = {
                                'Phase': ['Retrieval', 'Generation'],
                                'Time (ms)': [perf['retrieval_time_ms'], perf['generation_time_ms']]
                            }
                            
                            fig = px.bar(
                                perf_data, 
                                x='Phase', 
                                y='Time (ms)',
                                title="Performance Breakdown",
                                color='Phase'
                            )
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Quality assessment
                            if result.get("quality_assessment"):
                                st.subheader("Quality Assessment")
                                quality = result["quality_assessment"]
                                
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("Overall Quality", f"{quality['overall_quality']:.2f}")
                                    st.metric("Relevance", f"{quality['relevance_score']:.2f}")
                                with col2:
                                    st.metric("Coherence", f"{quality['coherence_score']:.2f}")
                                    st.metric("Completeness", f"{quality['completeness_score']:.2f}")
                                with col3:
                                    st.metric("Factual Accuracy", f"{quality['factual_accuracy']:.2f}")
                                    st.metric("Context Utilization", f"{quality['context_utilization']:.2f}")
                                
                                # Quality radar chart
                                fig = go.Figure()
                                
                                categories = ['Relevance', 'Coherence', 'Factual Accuracy', 
                                            'Completeness', 'Context Utilization']
                                values = [
                                    quality['relevance_score'],
                                    quality['coherence_score'],
                                    quality['factual_accuracy'],
                                    quality['completeness_score'],
                                    quality['context_utilization']
                                ]
                                
                                fig.add_trace(go.Scatterpolar(
                                    r=values,
                                    theta=categories,
                                    fill='toself',
                                    name='Quality Metrics'
                                ))
                                
                                fig.update_layout(
                                    polar=dict(
                                        radialaxis=dict(
                                            visible=True,
                                            range=[0, 1]
                                        )),
                                    showlegend=True,
                                    title="Quality Assessment Radar"
                                )
                                
                                st.plotly_chart(fig, use_container_width=True)
                            
                            # Session details
                            with st.expander("Detailed Metrics"):
                                st.json(result)
                            
                            # Word Export Section
                            st.markdown("---")
                            st.subheader("Export Assessment Results")
                            if st.button("Export to Word", type="secondary", key="export_rag_assessment"):
                                try:
                                    with st.spinner("Generating Word document..."):
                                        # Call FastAPI export endpoint
                                        export_response = requests.post(
                                            f"{FASTAPI_API}/export-rag-assessment-word",
                                            json=result,
                                            timeout=30
                                        )
                                        
                                        if export_response.status_code == 200:
                                            response_data = export_response.json()
                                            file_content = response_data.get("content_b64")
                                            filename = response_data.get("filename", "rag_assessment_export.docx")
                                            
                                            if file_content:
                                                import base64
                                                doc_bytes = base64.b64decode(file_content)
                                                
                                                st.download_button(
                                                    label=f"Download {filename}",
                                                    data=doc_bytes,
                                                    file_name=filename,
                                                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                                                    key="download_rag_assessment"
                                                )
                                                st.success("RAG assessment exported successfully!")
                                            else:
                                                st.error("No file content received")
                                        else:
                                            st.error(f"Export failed: {export_response.text}")
                                except Exception as e:
                                    st.error(f"Error exporting RAG assessment: {str(e)}")
                        
                        else:
                            st.error(f"Assessment failed ({response.status_code}): {response.text}")
                    
                    except Exception as e:
                        st.error(f"Assessment error: {str(e)}")
    
    # ===== ANALYTICS TAB =====
    with analytics_tab:
        st.subheader("Performance Analytics")
        st.info("View comprehensive performance analytics across time periods.")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            time_period = st.selectbox(
                "Analysis Period:",
                [1, 6, 12, 24, 48, 168],  # hours
                format_func=lambda x: f"{x} hours" if x < 24 else f"{x//24} days",
                index=3,  # default to 24 hours
                key="analytics_time_period"
            )
        
        with col2:
            analytics_collection = st.selectbox(
                "Filter by Collection:",
                ["All Collections"] + (collections if collections else []),
                key="analytics_collection_filter"
            )
        
        if st.button("Generate Analytics Report", key="generate_analytics"):
            with st.spinner("Generating comprehensive analytics report..."):
                try:
                    payload = {
                        "time_period_hours": time_period,
                        "collection_name": analytics_collection if analytics_collection != "All Collections" else None
                    }
                    
                    response = requests.post(
                        f"{FASTAPI_API}/rag-analytics",
                        json=payload,
                        timeout=60
                    )
                    
                    if response.status_code == 200:
                        analytics = response.json()
                        
                        if "message" in analytics:
                            st.warning(analytics["message"])
                        else:
                            st.success("Analytics Report Generated!")
                            
                            # Overview metrics
                            st.subheader("Overview")
                            query_stats = analytics["query_statistics"]
                            
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("Total Queries", query_stats["total_queries"])
                            with col2:
                                st.metric("Success Rate", f"{query_stats['success_rate']:.1%}")
                            with col3:
                                st.metric("Successful", query_stats["successful_queries"])
                            with col4:
                                st.metric("Failed", query_stats["failed_queries"])
                            
                            # Performance metrics
                            st.subheader("Performance Metrics")
                            perf_metrics = analytics["performance_metrics"]
                            
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric(
                                    "Avg Response Time", 
                                    f"{perf_metrics['response_time_ms']['mean']:.1f}ms"
                                )
                                st.metric(
                                    "Median Response Time", 
                                    f"{perf_metrics['response_time_ms']['median']:.1f}ms"
                                )
                            with col2:
                                st.metric(
                                    "Avg Retrieval Time", 
                                    f"{perf_metrics['retrieval_time_ms']['mean']:.1f}ms"
                                )
                                st.metric(
                                    "Avg Generation Time", 
                                    f"{perf_metrics['generation_time_ms']['mean']:.1f}ms"
                                )
                            with col3:
                                st.metric(
                                    "Min Response Time", 
                                    f"{perf_metrics['response_time_ms']['min']:.1f}ms"
                                )
                                st.metric(
                                    "Max Response Time", 
                                    f"{perf_metrics['response_time_ms']['max']:.1f}ms"
                                )
                            
                            # Usage patterns
                            st.subheader("🔍 Usage Patterns")
                            usage = analytics["usage_patterns"]
                            
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.write("**Collections Used:**")
                                for collection in usage["collections_used"]:
                                    st.write(f"• {collection}")
                                
                                st.metric("Avg Documents Retrieved", f"{usage['avg_documents_retrieved']:.1f}")
                            
                            with col2:
                                st.write("**Models Used:**")
                                for model in usage["models_used"]:
                                    st.write(f"• {model}")
                                
                                st.metric("Avg Response Length", f"{usage['avg_response_length']:.0f} chars")
                            
                            # Quality metrics (if available)
                            if "quality_metrics" in analytics and analytics["quality_metrics"]["overall_quality"]["samples"] > 0:
                                st.subheader("Quality Metrics")
                                quality = analytics["quality_metrics"]["overall_quality"]
                                
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("Avg Quality Score", f"{quality['mean']:.2f}")
                                with col2:
                                    st.metric("Median Quality", f"{quality['median']:.2f}")
                                with col3:
                                    st.metric("Quality Samples", quality['samples'])
                            
                            # Raw analytics data
                            with st.expander("📊 Raw Analytics Data"):
                                st.json(analytics)
                    
                    else:
                        st.error(f"Analytics failed ({response.status_code}): {response.text}")
                
                except Exception as e:
                    st.error(f"Analytics error: {str(e)}")
    
    # ===== BENCHMARKING TAB =====
    with benchmark_tab:
        st.subheader("RAG Configuration Benchmarking")
        st.info("Compare different RAG configurations on standardized query sets.")
        
        # Benchmark setup
        st.write("**Step 1: Define Test Queries**")
        benchmark_queries = st.text_area(
            "Test Queries (one per line):",
            placeholder="What are the compliance requirements?\nAnalyze contract risks\nSummarize key terms",
            height=100,
            key="benchmark_queries"
        )
        
        st.write("**Step 2: Select Collection**")
        if collections:
            benchmark_collection = st.selectbox(
                "Benchmark Collection:",
                collections,
                key="benchmark_collection"
            )
        else:
            st.warning("No collections available for benchmarking")
            benchmark_collection = None
        
        st.write("**Step 3: Configure Test Scenarios**")
        
        # Configuration builder
        configs = []
        
        config_expander = st.expander("➕ Configuration Builder", expanded=True)
        with config_expander:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                config_model = st.selectbox(
                    "Model:",
                    ["gpt-4", "gpt-3.5-turbo"],
                    key="config_model"
                )
            
            with col2:
                config_top_k = st.slider("Top K:", 1, 20, 5, key="config_top_k")
            
            with col3:
                if st.button("➕ Add Configuration", key="add_config"):
                    new_config = {
                        "model_name": config_model,
                        "top_k": config_top_k
                    }
                    if "benchmark_configs" not in st.session_state:
                        st.session_state.benchmark_configs = []
                    st.session_state.benchmark_configs.append(new_config)
                    st.success(f"Added: {config_model} (top_k={config_top_k})")
                    st.rerun()
        
        # Display configured scenarios
        if "benchmark_configs" in st.session_state and st.session_state.benchmark_configs:
            st.write("**Configured Test Scenarios:**")
            for i, config in enumerate(st.session_state.benchmark_configs):
                col1, col2 = st.columns([4, 1])
                with col1:
                    st.write(f"**Config {i+1}:** {config['model_name']} (top_k={config['top_k']})")
                with col2:
                    if st.button("X", key=f"remove_config_{i}", help="Remove this configuration"):
                        st.session_state.benchmark_configs.pop(i)
                        st.rerun()
        
        # Run benchmark
        if st.button("Run Benchmark", type="primary", key="run_benchmark"):
            if not benchmark_queries:
                st.warning("Please provide test queries.")
            elif not benchmark_collection:
                st.warning("Please select a collection.")
            elif "benchmark_configs" not in st.session_state or not st.session_state.benchmark_configs:
                st.warning("Please add at least one configuration.")
            else:
                query_list = [q.strip() for q in benchmark_queries.split('\n') if q.strip()]
                
                with st.spinner(f"Running benchmark with {len(query_list)} queries across {len(st.session_state.benchmark_configs)} configurations..."):
                    try:
                        payload = {
                            "query_set": query_list,
                            "collection_name": benchmark_collection,
                            "configurations": st.session_state.benchmark_configs
                        }
                        
                        response = requests.post(
                            f"{FASTAPI_API}/rag-benchmark",
                            json=payload,
                            timeout=600  # 10 minutes for benchmarking
                        )
                        
                        if response.status_code == 200:
                            results = response.json()
                            
                            st.success("Benchmark Complete!")
                            
                            # Results overview
                            st.subheader("Benchmark Results")
                            
                            benchmark_data = []
                            for config_id, result in results["results"].items():
                                config = result["configuration"]
                                metrics = result["results"]
                                
                                benchmark_data.append({
                                    "Configuration": f"{config['model_name']} (k={config['top_k']})",
                                    "Avg Response Time (ms)": metrics["avg_response_time_ms"],
                                    "Avg Quality Score": metrics["avg_quality_score"],
                                    "Queries Tested": metrics["queries_tested"]
                                })
                            
                            # Results table
                            df = pd.DataFrame(benchmark_data)
                            st.dataframe(df, use_container_width=True)
                            
                            # Performance comparison chart
                            fig = px.scatter(
                                df, 
                                x="Avg Response Time (ms)", 
                                y="Avg Quality Score",
                                text="Configuration",
                                title="Performance vs Quality Trade-off",
                                hover_data=["Queries Tested"]
                            )
                            fig.update_traces(textposition="top center")
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Detailed results
                            with st.expander("📊 Detailed Benchmark Results"):
                                st.json(results)
                        
                        else:
                            st.error(f"Benchmark failed ({response.status_code}): {response.text}")
                    
                    except Exception as e:
                        st.error(f"Benchmark error: {str(e)}")
    
    # ===== COLLECTIONS TAB =====
    with collections_tab:
        st.subheader("Collection Performance Analysis")
        st.info("Analyze RAG performance for specific document collections.")
        
        if collections:
            selected_collection = st.selectbox(
                "Select Collection for Analysis:",
                collections,
                key="collection_analysis_select"
            )
            
            if st.button("Analyze Collection", key="analyze_collection"):
                with st.spinner(f"Analyzing performance for collection: {selected_collection}"):
                    try:
                        response = requests.get(
                            f"{FASTAPI_API}/rag-collection-performance/{selected_collection}",
                            timeout=60
                        )
                        
                        if response.status_code == 200:
                            perf_data = response.json()
                            
                            if "message" in perf_data:
                                st.warning(perf_data["message"])
                            else:
                                st.success(f"Collection Analysis: {selected_collection}")
                                
                                # Collection metrics
                                col1, col2, col3, col4 = st.columns(4)
                                with col1:
                                    st.metric("Total Queries", perf_data["total_queries"])
                                with col2:
                                    st.metric("Success Rate", f"{perf_data['success_rate']:.1%}")
                                with col3:
                                    st.metric("Avg Response Time", f"{perf_data['avg_response_time_ms']:.1f}ms")
                                with col4:
                                    st.metric("Avg Docs Retrieved", f"{perf_data['avg_documents_retrieved']:.1f}")
                                
                                # Recent queries
                                if perf_data.get("recent_queries"):
                                    st.subheader("Recent Queries")
                                    
                                    recent_df = pd.DataFrame(perf_data["recent_queries"])
                                    recent_df["timestamp"] = pd.to_datetime(recent_df["timestamp"])
                                    recent_df = recent_df.sort_values("timestamp", ascending=False)
                                    
                                    st.dataframe(recent_df, use_container_width=True)
                                
                                # Raw data
                                with st.expander("Raw Collection Data"):
                                    st.json(perf_data)
                        
                        else:
                            st.error(f"Collection analysis failed ({response.status_code}): {response.text}")
                    
                    except Exception as e:
                        st.error(f"Collection analysis error: {str(e)}")
        else:
            st.warning("No collections available for analysis.")
    
    # ===== EXPORT TAB =====
    with export_tab:
        st.subheader("Export & Reports")
        st.info("Export RAG metrics and generate comprehensive reports for external analysis.")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            export_period = st.selectbox(
                "Export Period:",
                [1, 6, 12, 24, 48, 168, 720],  # hours
                format_func=lambda x: f"{x} hours" if x < 24 else f"{x//24} days",
                index=3,  # default to 24 hours
                key="export_time_period"
            )
        
        with col2:
            export_format = st.selectbox(
                "Export Format:",
                ["json", "csv"],
                key="export_format"
            )
        
        if st.button("Export Metrics", key="export_metrics"):
            with st.spinner("Preparing metrics export..."):
                try:
                    payload = {
                        "format": export_format,
                        "time_period_hours": export_period
                    }
                    
                    response = requests.post(
                        f"{FASTAPI_API}/rag-export-metrics",
                        json=payload,
                        timeout=60
                    )
                    
                    if response.status_code == 200:
                        export_data = response.json()
                        
                        st.success("Export Generated!")
                        
                        # Export summary
                        summary = export_data["summary"]
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("Total Sessions", summary["total_sessions"])
                        with col2:
                            st.metric("Successful Sessions", summary["successful_sessions"])
                        with col3:
                            st.metric("Quality Assessments", summary["quality_assessments_count"])
                        
                        # Download options
                        st.subheader("Download Options")
                        
                        # JSON download
                        json_str = json.dumps(export_data, indent=2, default=str)
                        st.download_button(
                            label="Download JSON Export",
                            data=json_str,
                            file_name=f"rag_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                            mime="application/json",
                            key="download_json"
                        )
                        
                        # Preview
                        with st.expander("Preview Export Data"):
                            st.json(export_data)
                    
                    else:
                        st.error(f"Export failed ({response.status_code}): {response.text}")
                
                except Exception as e:
                    st.error(f"Export error: {str(e)}")
        
        # Demo and documentation
        st.markdown("---")
        st.subheader("RAG Assessment Documentation")
        
        if st.button("View API Documentation", key="view_docs"):
            with st.spinner("Loading documentation..."):
                try:
                    response = requests.get(f"{FASTAPI_API}/rag-assessment-demo", timeout=10)
                    if response.status_code == 200:
                        docs = response.json()
                        
                        st.json(docs)
                    else:
                        st.error("Documentation not available")
                except Exception as e:
                    st.error(f"Documentation error: {str(e)}")