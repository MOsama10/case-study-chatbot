
# # ### src/ui.py
# # """
# # Gradio web interface for the case study chatbot.
# # """
# # from typing import Dict, Any, List, Tuple, Optional
# # import gradio as gr
# # import logging
# # import json
# # from pathlib import Path

# # from src.config import get_logger, GRADIO_PORT, GRADIO_SHARE, DATA_ROOT
# # from src.chatbot import handle_message, get_session_info, clear_user_session
# # from src.embeddings import build_and_save_embeddings, load_index
# # from src.knowledge_graph import load_kg, build_kg, save_kg
# # from src.data_loader import load_docx

# # logger = get_logger(__name__)


# # class ChatbotUI:
# #     """Gradio UI for the case study chatbot."""
    
# #     def __init__(self):
# #         """Initialize the UI."""
# #         self.current_user = "default_user"
# #         self.setup_complete = False
        
# #         # Initialize system
# #         self._initialize_system()
    
# #     def _initialize_system(self) -> None:
# #         """Initialize embeddings and knowledge graph."""
# #         try:
# #             logger.info("Initializing chatbot system...")
            
# #             # Try to load existing indexes
# #             embeddings_loaded = load_index()
# #             kg_loaded = load_kg()
            
# #             if not embeddings_loaded or not kg_loaded:
# #                 logger.info("Building new indexes from documents...")
# #                 self._build_indexes()
            
# #             self.setup_complete = True
# #             logger.info("System initialization complete")
            
# #         except Exception as e:
# #             logger.error(f"System initialization failed: {e}")
# #             self.setup_complete = False
    
# #     def _build_indexes(self) -> None:
# #         """Build embeddings and knowledge graph from documents."""
# #         doc_path = DATA_ROOT / "cases.docx"
        
# #         if doc_path.exists():
# #             # Load and process document
# #             items = load_docx(doc_path)
# #             logger.info(f"Loaded {len(items)} items from document")
            
# #             # Build embeddings
# #             build_and_save_embeddings(doc_path)
            
# #             # Build knowledge graph
# #             from src.knowledge_graph import build_kg
# #             build_kg(items)
# #             save_kg()
            
# #         else:
# #             # Build with sample data
# #             logger.warning("No document found, building with sample data")
# #             build_and_save_embeddings()
    
# #     def chat_interface(self, message: str, history: List[List[str]]) -> Tuple[str, List[List[str]]]:
# #         """
# #         Handle chat interface interactions.
        
# #         Args:
# #             message: User message
# #             history: Chat history
            
# #         Returns:
# #             Tuple of (empty string, updated history)
# #         """
# #         if not self.setup_complete:
# #             return "", history + [[message, "⚠️ System is still initializing. Please wait..."]]
        
# #         if not message.strip():
# #             return "", history
        
# #         try:
# #             # Get response from chatbot
# #             response = handle_message(self.current_user, message.strip())
            
# #             # Format response with sources
# #             formatted_response = self._format_response(response)
            
# #             # Update history
# #             new_history = history + [[message, formatted_response]]
            
# #             return "", new_history
            
# #         except Exception as e:
# #             logger.error(f"Chat interface error: {e}")
# #             error_response = f"❌ Sorry, I encountered an error: {str(e)}"
# #             return "", history + [[message, error_response]]
    
# #     def _format_response(self, response: Dict) -> str:
# #         """Format the chatbot response for display with direct answer handling."""
# #         answer = response['answer']
# #         sources = response['sources']
# #         confidence = response['confidence']
# #         processing_time = response['processing_time']
# #         query_type = response.get('query_type', 'general')
        
# #         # Handle direct answers differently
# #         if query_type == 'direct_answer':
# #             # More concise formatting for direct answers
# #             formatted = answer
            
# #             # Add brief metadata for direct answers
# #             formatted += f"\n\n---"
# #             formatted += f"\n**Quick Answer Metrics:**"
# #             formatted += f"\n• Confidence: {confidence:.0%}"
# #             formatted += f"\n• Sources: {len(sources)} case studies"
# #             formatted += f"\n• Response time: {processing_time:.1f}s"
            
# #             # Show top 2 sources for direct answers
# #             if sources:
# #                 formatted += f"\n\n**Key References:**"
# #                 for i, source in enumerate(sources[:2], 1):
# #                     source_text = source['text'][:100] + "..." if len(source['text']) > 100 else source['text']
# #                     formatted += f"\n{i}. {source_text}"
            
# #             return formatted
        
# #         else:
# #             # Comprehensive formatting for detailed analysis
# #             formatted = answer
            
# #             # Add detailed metadata for comprehensive responses
# #             formatted += f"\n\n---"
# #             formatted += f"\n**Business Analysis Metrics:**"
# #             formatted += f"\n• Analysis Confidence: {confidence:.0%}"
# #             formatted += f"\n• Processing Time: {processing_time:.2f}s"
# #             formatted += f"\n• Evidence Sources: {len(sources)} case studies"
# #             formatted += f"\n• Response Type: {query_type.title().replace('_', ' ')}"
            
# #             # Add comprehensive source information
# #             if sources:
# #                 formatted += f"\n\n**Case Study References:**"
# #                 for i, source in enumerate(sources[:3], 1):
# #                     source_type = source.get('type', 'Case Study')
# #                     score = source.get('score', 0)
# #                     formatted += f"\n{i}. [{source_type.upper()}] Relevance: {score:.0%}"
            
# #             # Add KG info if available
# #             kg_nodes = response.get('kg_nodes', [])
# #             if kg_nodes:
# #                 formatted += f"\n• Knowledge connections: {len(kg_nodes)} found"
            
# #             return formatted
    
# #     def clear_conversation(self) -> List[List[str]]:
# #         """Clear the current conversation."""
# #         clear_user_session(self.current_user)
# #         logger.info(f"Cleared conversation for user {self.current_user}")
# #         return []
    
# #     def get_session_stats(self) -> str:
# #         """Get current session statistics."""
# #         session_info = get_session_info(self.current_user)
        
# #         if session_info:
# #             stats = f"""**Session Statistics:**
# # - Total turns: {session_info['total_turns']}
# # - Average confidence: {session_info.get('avg_confidence', 0):.1%}
# # - Average response time: {session_info.get('avg_processing_time', 0):.2f}s
# # - Query types: {', '.join(session_info.get('query_types', []))}"""
# #         else:
# #             stats = "No active session"
        
# #         return stats
    
# #     def create_interface(self) -> gr.Blocks:
# #         """Create the Gradio interface."""
        
# #         # Custom CSS for better styling
# #         custom_css = """
# #         .gradio-container {
# #             max-width: 1200px !important;
# #         }
        
# #         .chat-container {
# #             height: 500px;
# #             overflow-y: auto;
# #         }
        
# #         .source-box {
# #             background-color: #f0f0f0;
# #             padding: 10px;
# #             border-radius: 5px;
# #             margin: 5px 0;
# #         }
# #         """
        
# #         with gr.Blocks(
# #             theme=gr.themes.Soft(),
# #             css=custom_css,
# #             title="Case Study Chatbot"
# #         ) as interface:
            
# #             # Header
# #             gr.Markdown(
# #                 """
# #                 # 📊 Case Study Conversational Chatbot
                
# #                 Welcome to the AI-powered case study analysis assistant. Ask questions about business cases, 
# #                 request analysis, or explore insights from the knowledge base.
                
# #                 **Example queries:**
# #                 - "What are the main customer service problems?"
# #                 - "How did companies solve their turnover issues?"
# #                 - "Analyze the effectiveness of training programs"
# #                 """
# #             )
            
# #             # Status indicator
# #             with gr.Row():
# #                 status_text = "🟢 System Ready" if self.setup_complete else "🟡 Initializing..."
# #                 gr.Markdown(f"**Status:** {status_text}")
            
# #             # Main chat interface
# #             with gr.Row():
# #                 with gr.Column(scale=3):
# #                     chatbot = gr.Chatbot(
# #                         label="Conversation",
# #                         height=500,
# #                         show_label=True,
# #                         container=True,
# #                         show_copy_button=True
# #                     )
                    
# #                     msg = gr.Textbox(
# #                         label="Your message",
# #                         placeholder="Ask me about case studies...",
# #                         container=True,
# #                         scale=4
# #                     )
                    
# #                     with gr.Row():
# #                         submit_btn = gr.Button("Send", variant="primary", scale=1)
# #                         clear_btn = gr.Button("Clear Chat", scale=1)
                
# #                 # Sidebar with info
# #                 with gr.Column(scale=1):
# #                     gr.Markdown("### 📈 Session Info")
                    
# #                     stats_display = gr.Markdown("No active session")
                    
# #                     refresh_stats_btn = gr.Button("Refresh Stats", size="sm")
                    
# #                     gr.Markdown("### 💡 Tips")
# #                     gr.Markdown(
# #                         """
# #                         - Ask specific questions for better results
# #                         - Request analysis for deeper insights
# #                         - Sources are automatically cited
# #                         - Use "Clear Chat" to start over
# #                         """
# #                     )
                    
# #                     # System info
# #                     gr.Markdown("### ⚙️ System Info")
# #                     system_info = f"""
# #                     - **LLM Provider:** {self._get_llm_provider()}
# #                     - **Embedding Model:** all-MiniLM-L6-v2
# #                     - **Vector Index:** FAISS
# #                     - **Knowledge Graph:** NetworkX
# #                     """
# #                     gr.Markdown(system_info)
            
# #             # Event handlers
# #             msg.submit(
# #                 self.chat_interface,
# #                 inputs=[msg, chatbot],
# #                 outputs=[msg, chatbot]
# #             )
            
# #             submit_btn.click(
# #                 self.chat_interface,
# #                 inputs=[msg, chatbot],
# #                 outputs=[msg, chatbot]
# #             )
            
# #             clear_btn.click(
# #                 self.clear_conversation,
# #                 outputs=[chatbot]
# #             )
            
# #             refresh_stats_btn.click(
# #                 self.get_session_stats,
# #                 outputs=[stats_display]
# #             )
            
# #             # Examples
# #             gr.Examples(
# #                 examples=[
# #                     "What are the main problems companies face?",
# #                     "How can businesses improve customer satisfaction?",
# #                     "Analyze the effectiveness of employee training programs",
# #                     "What solutions work best for reducing costs?",
# #                     "Find examples of successful problem-solving approaches"
# #                 ],
# #                 inputs=msg
# #             )
            
# #             # Footer
# #             gr.Markdown(
# #                 """
# #                 ---
# #                 **Case Study Chatbot PoC** | Built with Gradio, FAISS, NetworkX, and LLM APIs
# #                 """
# #             )
        
# #         return interface
    
# #     def _get_llm_provider(self) -> str:
# #         """Get current LLM provider name."""
# #         from src.config import LLM_PROVIDER
# #         return LLM_PROVIDER.title()
    
# #     def launch(self, **kwargs) -> None:
# #         """Launch the Gradio interface."""
# #         interface = self.create_interface()
        
# #         launch_kwargs = {
# #             'server_port': GRADIO_PORT,
# #             'share': GRADIO_SHARE,
# #             'show_error': True,
# #             'quiet': False,
# #             **kwargs
# #         }
        
# #         logger.info(f"Launching Gradio interface on port {GRADIO_PORT}")
# #         interface.launch(**launch_kwargs)


# # def main():
# #     """Main entry point for the UI."""
# #     import sys
    
# #     # Configure logging for UI
# #     logging.basicConfig(
# #         level=logging.INFO,
# #         format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
# #         handlers=[
# #             logging.StreamHandler(sys.stdout),
# #         ]
# #     )
    
# #     # Create and launch UI
# #     ui = ChatbotUI()
    
# #     try:
# #         ui.launch()
# #     except KeyboardInterrupt:
# #         logger.info("Shutting down chatbot UI")
# #     except Exception as e:
# #         logger.error(f"Error running UI: {e}")
# #         sys.exit(1)


# # if __name__ == "__main__":
# #     main()

# """
# Enhanced Gradio web interface for the case study chatbot with creative design.
# """
# from typing import Dict, Any, List, Tuple, Optional
# import gradio as gr
# import logging
# import json
# import time
# from pathlib import Path
# import random

# from src.config import get_logger, GRADIO_PORT, GRADIO_SHARE, DATA_ROOT
# from src.chatbot import handle_message, get_session_info, clear_user_session
# from src.embeddings import build_and_save_embeddings, load_index
# from src.knowledge_graph import load_kg, build_kg, save_kg
# from src.data_loader import load_docx

# logger = get_logger(__name__)


# class EnhancedChatbotUI:
#     """Enhanced Gradio UI for the case study chatbot with creative design."""
    
#     def __init__(self):
#         """Initialize the enhanced UI."""
#         self.current_user = "default_user"
#         self.setup_complete = False
#         self.chat_history = []
#         self.typing_delay = 0.02  # For typing animation effect
        
#         # Creative elements
#         self.business_quotes = [
#             "💡 'Innovation distinguishes between a leader and a follower.' - Steve Jobs",
#             "📈 'Quality is not an act, it is a habit.' - Aristotle", 
#             "🎯 'The customer's perception is your reality.' - Kate Zabriskie",
#             "🚀 'Success is not final, failure is not fatal: it is the courage to continue that counts.' - Winston Churchill",
#             "⚡ 'Efficiency is doing things right; effectiveness is doing the right things.' - Peter Drucker"
#         ]
        
#         self.example_queries = [
#             ("🎯 Customer Service", "What are the most effective strategies for improving customer service response times?"),
#             ("👥 Employee Retention", "How can companies reduce employee turnover and improve job satisfaction?"),
#             ("💰 Cost Reduction", "What proven methods help companies reduce operational costs without affecting quality?"),
#             ("📊 Digital Transformation", "What are the key success factors for digital transformation initiatives?"),
#             ("🎓 Training Programs", "How do successful companies design and implement effective training programs?"),
#             ("📈 Quality Management", "What quality control measures have proven most effective in manufacturing?"),
#             ("💡 Innovation", "How do companies foster innovation and accelerate product development?"),
#             ("📋 Strategic Analysis", "Analyze the common patterns in successful business transformation cases"),
#         ]
        
#         # Initialize system
#         self._initialize_system()
    
#     def _initialize_system(self) -> None:
#         """Initialize embeddings and knowledge graph."""
#         try:
#             logger.info("🚀 Initializing chatbot system...")
            
#             # Try to load existing indexes
#             embeddings_loaded = load_index()
            
#             try:
#                 kg_loaded = load_kg()
#             except Exception as e:
#                 logger.warning(f"KG loading failed: {e}")
#                 kg_loaded = False
            
#             if not embeddings_loaded or not kg_loaded:
#                 logger.info("📚 Building new indexes from documents...")
#                 self._build_indexes()
            
#             self.setup_complete = True
#             logger.info("✅ System initialization complete")
            
#         except Exception as e:
#             logger.error(f"❌ System initialization failed: {e}")
#             self.setup_complete = False
    
#     def _build_indexes(self) -> None:
#         """Build embeddings and knowledge graph from documents."""
#         # Check for existing documents first
#         doc_paths = [
#             DATA_ROOT / "cases.docx",
#             DATA_ROOT / "batch_1" / "advanced_cases.docx"
#         ]
        
#         existing_docs = [p for p in doc_paths if p.exists()]
        
#         if not existing_docs:
#             # Create sample data if no documents exist
#             logger.info("📝 No documents found, creating sample data...")
#             try:
#                 from create_sample_data import create_sample_case_studies, create_batch_data
#                 create_sample_case_studies()
#                 create_batch_data()
#                 existing_docs = [p for p in doc_paths if p.exists()]
#             except ImportError:
#                 logger.warning("Could not create sample data - using fallback")
        
#         if existing_docs:
#             # Build with first available document
#             build_and_save_embeddings(existing_docs[0])
#         else:
#             # Build with sample data
#             build_and_save_embeddings()
        
#         # Build knowledge graph
#         try:
#             kg = build_kg()
#             save_kg(kg)
#         except Exception as e:
#             logger.warning(f"KG building failed: {e}")
    
#     def chat_interface(self, message: str, history: List[List[str]]) -> Tuple[str, List[List[str]]]:
#         """
#         Handle chat interface interactions with enhanced formatting.
#         """
#         if not self.setup_complete:
#             return "", history + [[message, "⚠️ System is still initializing... Please wait a moment and try again."]]
        
#         if not message.strip():
#             return "", history
        
#         try:
#             # Add loading indicator
#             loading_response = "🤔 Analyzing your question..."
#             temp_history = history + [[message, loading_response]]
            
#             # Get response from chatbot
#             response = handle_message(self.current_user, message.strip())
            
#             # Format response with enhanced styling
#             formatted_response = self._format_enhanced_response(response, message)
            
#             # Update history with final response
#             new_history = history + [[message, formatted_response]]
            
#             return "", new_history
            
#         except Exception as e:
#             logger.error(f"Chat interface error: {e}")
#             error_response = f"❌ Oops! I encountered an error: {str(e)}\n\n💡 Try rephrasing your question or check the system status."
#             return "", history + [[message, error_response]]
    
#     def _format_enhanced_response(self, response: Dict, original_query: str) -> str:
#         """Format the chatbot response with enhanced styling and creativity."""
#         answer = response['answer']
#         sources = response['sources']
#         confidence = response['confidence']
#         processing_time = response['processing_time']
#         query_type = response.get('query_type', 'general')
        
#         # Start with the main answer
#         formatted = answer
        
#         # Add creative confidence indicator
#         confidence_emoji = self._get_confidence_emoji(confidence)
#         confidence_text = self._get_confidence_text(confidence)
        
#         # Enhanced metadata section with better design
#         formatted += f"\n\n---\n"
#         formatted += f"## 📊 Analysis Insights\n\n"
        
#         # Confidence with visual indicator
#         formatted += f"**{confidence_emoji} Confidence Level:** {confidence_text} ({confidence:.0%})\n"
        
#         # Processing info with performance indicator
#         speed_emoji = "⚡" if processing_time < 3 else "🐌" if processing_time > 10 else "🚀"
#         formatted += f"**{speed_emoji} Processing Time:** {processing_time:.2f}s\n"
        
#         # Query analysis
#         query_emoji = self._get_query_type_emoji(query_type)
#         formatted += f"**{query_emoji} Analysis Type:** {query_type.title().replace('_', ' ')}\n"
        
#         # Evidence strength
#         evidence_emoji = "📚" if len(sources) > 3 else "📖" if len(sources) > 1 else "📄"
#         formatted += f"**{evidence_emoji} Evidence Sources:** {len(sources)} case studies analyzed\n"
        
#         # Add source information with better formatting
#         if sources:
#             formatted += f"\n## 🔍 Evidence Details\n\n"
            
#             for i, source in enumerate(sources[:3], 1):  # Top 3 sources
#                 relevance = source.get('score', 0)
#                 source_type = source.get('type', 'Case Study')
                
#                 # Relevance indicator
#                 relevance_emoji = "🎯" if relevance > 0.8 else "📌" if relevance > 0.6 else "📍"
                
#                 formatted += f"**{relevance_emoji} Source {i}** ({source_type.upper()}) - Relevance: {relevance:.0%}\n"
                
#                 # Source preview with smart truncation
#                 text_preview = source.get('text', '')
#                 if len(text_preview) > 120:
#                     # Try to cut at sentence boundary
#                     sentences = text_preview.split('. ')
#                     preview = sentences[0]
#                     if len(preview) < 80 and len(sentences) > 1:
#                         preview += '. ' + sentences[1]
#                     preview = preview[:120] + "..."
#                 else:
#                     preview = text_preview
                
#                 formatted += f"*{preview}*\n\n"
        
#         # Add interactive elements based on query type
#         if query_type in ['analysis', 'recommendation']:
#             formatted += self._add_analysis_suggestions(original_query)
#         elif query_type == 'direct_answer':
#             formatted += self._add_follow_up_suggestions(original_query)
        
#         # Add footer with inspirational element
#         formatted += f"\n---\n"
#         formatted += f"💼 *{random.choice(self.business_quotes)}*\n"
#         formatted += f"⏰ *Generated at {time.strftime('%H:%M:%S')}*"
        
#         return formatted
    
#     def _get_confidence_emoji(self, confidence: float) -> str:
#         """Get emoji based on confidence level."""
#         if confidence >= 0.9:
#             return "🎯"
#         elif confidence >= 0.8:
#             return "✅"
#         elif confidence >= 0.7:
#             return "👍"
#         elif confidence >= 0.6:
#             return "⚖️"
#         elif confidence >= 0.5:
#             return "🤔"
#         else:
#             return "⚠️"
    
#     def _get_confidence_text(self, confidence: float) -> str:
#         """Get confidence level description."""
#         if confidence >= 0.9:
#             return "Excellent"
#         elif confidence >= 0.8:
#             return "High"
#         elif confidence >= 0.7:
#             return "Good"
#         elif confidence >= 0.6:
#             return "Moderate"
#         elif confidence >= 0.5:
#             return "Fair"
#         else:
#             return "Limited"
    
#     def _get_query_type_emoji(self, query_type: str) -> str:
#         """Get emoji for query type."""
#         emoji_map = {
#             'analysis': '📊',
#             'recommendation': '💡', 
#             'comparison': '⚖️',
#             'trend': '📈',
#             'implementation': '🚀',
#             'direct_answer': '🎯',
#             'qa': '❓',
#             'search': '🔍',
#             'summary': '📋'
#         }
#         return emoji_map.get(query_type, '🤖')
    
#     def _add_analysis_suggestions(self, query: str) -> str:
#         """Add follow-up analysis suggestions."""
#         suggestions = [
#             "💡 Want to explore implementation strategies for these insights?",
#             "📊 Need a deeper dive into the metrics and KPIs?",
#             "🎯 Interested in risk assessment for these approaches?",
#             "🚀 Ready to discuss change management strategies?"
#         ]
        
#         selected = random.choice(suggestions)
#         return f"\n**💭 Next Steps:** {selected}\n"
    
#     def _add_follow_up_suggestions(self, query: str) -> str:
#         """Add follow-up suggestions for direct answers."""
#         suggestions = [
#             "🔍 Want to see more detailed case studies on this topic?",
#             "📈 Curious about implementation timelines and budgets?", 
#             "⚖️ Need help comparing different approaches?",
#             "🎯 Ready for a comprehensive analysis?"
#         ]
        
#         selected = random.choice(suggestions)
#         return f"\n**🤝 Follow-up:** {selected}\n"
    
#     def clear_conversation(self) -> List[List[str]]:
#         """Clear the current conversation with confirmation."""
#         clear_user_session(self.current_user)
#         logger.info(f"🗑️ Cleared conversation for user {self.current_user}")
#         return []
    
#     def get_session_stats(self) -> str:
#         """Get current session statistics with enhanced formatting."""
#         session_info = get_session_info(self.current_user)
        
#         if session_info:
#             # Calculate performance indicators
#             avg_conf = session_info.get('avg_confidence', 0)
#             avg_time = session_info.get('avg_processing_time', 0)
            
#             conf_grade = "A+" if avg_conf > 0.8 else "A" if avg_conf > 0.7 else "B+" if avg_conf > 0.6 else "B"
#             speed_grade = "🚀 Fast" if avg_time < 3 else "⚡ Good" if avg_time < 6 else "🐌 Slow"
            
#             stats = f"""## 📊 Session Performance Dashboard

# ### 🎯 Quality Metrics
# - **Conversation Quality:** {conf_grade} ({avg_conf:.1%} avg confidence)
# - **Response Speed:** {speed_grade} ({avg_time:.2f}s avg)
# - **Total Interactions:** {session_info['total_turns']} exchanges

# ### 🧠 Analysis Breakdown  
# - **Query Types:** {', '.join(session_info.get('query_types', []))}
# - **Session Started:** {session_info.get('created_at', 'Unknown')}

# ### 💡 Insights
# {"🏆 Excellent session quality!" if avg_conf > 0.8 else "✅ Good analytical engagement!" if avg_conf > 0.6 else "📈 Keep exploring for better insights!"}"""
#         else:
#             stats = """## 🔄 New Session
            
# **Ready to start!** Ask me anything about business case studies, and I'll provide data-driven insights."""
        
#         return stats
    
#     def get_system_status(self) -> str:
#         """Get enhanced system status."""
#         status = "## 🔧 System Status\n\n"
        
#         if self.setup_complete:
#             status += "✅ **Core Systems:** All operational\n"
#             status += "✅ **Vector Search:** FAISS index loaded\n"
#             status += "✅ **Knowledge Graph:** NetworkX ready\n"
#             status += "✅ **LLM Provider:** Connected\n"
#             status += "✅ **Document Processing:** Active\n\n"
#             status += "🚀 **Ready for analysis!**"
#         else:
#             status += "⚠️ **Status:** Initializing...\n"
#             status += "🔄 **Progress:** Building indexes\n"
#             status += "⏱️ **ETA:** ~30 seconds\n\n"
#             status += "Please wait while I prepare the system."
        
#         return status
    
#     def create_interface(self) -> gr.Blocks:
#         """Create the enhanced Gradio interface with modern design."""
        
#         # Enhanced CSS with modern styling, animations, and dark/light theme support
#         enhanced_css = """
#         /* Modern UI Enhancements */
#         .gradio-container {
#             max-width: 1400px !important;
#             margin: 0 auto;
#             font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
#         }
        
#         /* Header styling */
#         .main-header {
#             background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
#             color: white;
#             padding: 2rem;
#             border-radius: 15px;
#             margin-bottom: 2rem;
#             box-shadow: 0 8px 32px rgba(102, 126, 234, 0.3);
#         }
        
#         /* Chat container enhancements */
#         .chat-container {
#             background: linear-gradient(145deg, #f8f9fa, #e9ecef);
#             border-radius: 20px;
#             padding: 1rem;
#             box-shadow: inset 0 4px 8px rgba(0,0,0,0.1);
#             min-height: 600px;
#         }
        
#         /* Message styling */
#         .message {
#             animation: slideIn 0.3s ease-out;
#             margin: 0.5rem 0;
#         }
        
#         @keyframes slideIn {
#             from { opacity: 0; transform: translateY(10px); }
#             to { opacity: 1; transform: translateY(0); }
#         }
        
#         /* Sidebar enhancements */
#         .sidebar {
#             background: rgba(255, 255, 255, 0.95);
#             backdrop-filter: blur(10px);
#             border-radius: 15px;
#             padding: 1.5rem;
#             box-shadow: 0 4px 20px rgba(0,0,0,0.1);
#         }
        
#         /* Button styling */
#         .primary-btn {
#             background: linear-gradient(45deg, #667eea, #764ba2);
#             color: white;
#             border: none;
#             border-radius: 25px;
#             padding: 0.8rem 2rem;
#             font-weight: 600;
#             transition: all 0.3s ease;
#             box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
#         }
        
#         .primary-btn:hover {
#             transform: translateY(-2px);
#             box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
#         }
        
#         /* Stats cards */
#         .stats-card {
#             background: rgba(255, 255, 255, 0.9);
#             border-radius: 10px;
#             padding: 1rem;
#             margin: 0.5rem 0;
#             border-left: 4px solid #667eea;
#             transition: transform 0.2s ease;
#         }
        
#         .stats-card:hover {
#             transform: translateX(5px);
#         }
        
#         /* Examples grid */
#         .examples-grid {
#             display: grid;
#             grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
#             gap: 1rem;
#             margin: 1rem 0;
#         }
        
#         .example-card {
#             background: linear-gradient(135deg, #ff9a9e 0%, #fecfef 100%);
#             border-radius: 15px;
#             padding: 1.5rem;
#             color: #333;
#             transition: all 0.3s ease;
#             cursor: pointer;
#             border: none;
#         }
        
#         .example-card:hover {
#             transform: translateY(-5px);
#             box-shadow: 0 10px 25px rgba(255, 154, 158, 0.3);
#         }
        
#         /* Status indicators */
#         .status-online {
#             color: #28a745;
#             font-weight: bold;
#         }
        
#         .status-loading {
#             color: #ffc107;
#             animation: pulse 1.5s infinite;
#         }
        
#         @keyframes pulse {
#             0%, 100% { opacity: 1; }
#             50% { opacity: 0.5; }
#         }
        
#         /* Responsive design */
#         @media (max-width: 768px) {
#             .gradio-container {
#                 max-width: 100% !important;
#                 margin: 0;
#                 padding: 0.5rem;
#             }
            
#             .main-header {
#                 padding: 1rem;
#             }
            
#             .examples-grid {
#                 grid-template-columns: 1fr;
#             }
#         }
        
#         /* Dark mode support */
#         @media (prefers-color-scheme: dark) {
#             .chat-container {
#                 background: linear-gradient(145deg, #2d3748, #4a5568);
#             }
            
#             .sidebar {
#                 background: rgba(45, 55, 72, 0.95);
#                 color: white;
#             }
            
#             .stats-card {
#                 background: rgba(45, 55, 72, 0.9);
#                 color: white;
#             }
#         }
#         """
        
#         with gr.Blocks(
#             theme=gr.themes.Soft(
#                 primary_hue="blue",
#                 secondary_hue="purple",
#                 neutral_hue="gray",
#                 radius_size="lg",
#                 spacing_size="lg"
#             ),
#             css=enhanced_css,
#             title="🚀 AI Business Case Study Analyzer"
#         ) as interface:
            
#             # Enhanced Header with gradient and animations
#             with gr.Row(elem_classes="main-header"):
#                 gr.HTML("""
#                 <div style="text-align: center;">
#                     <h1 style="font-size: 3rem; margin-bottom: 1rem; text-shadow: 2px 2px 4px rgba(0,0,0,0.3);">
#                         🚀 AI Business Case Study Analyzer
#                     </h1>
#                     <p style="font-size: 1.2rem; opacity: 0.9; max-width: 800px; margin: 0 auto;">
#                         Unlock powerful insights from business case studies using advanced AI analysis, 
#                         semantic search, and knowledge graphs. Get data-driven recommendations for your business challenges.
#                     </p>
#                     <div style="margin-top: 1rem;">
#                         <span style="background: rgba(255,255,255,0.2); padding: 0.5rem 1rem; border-radius: 20px; margin: 0 0.5rem;">
#                             🧠 GPT-Powered Analysis
#                         </span>
#                         <span style="background: rgba(255,255,255,0.2); padding: 0.5rem 1rem; border-radius: 20px; margin: 0 0.5rem;">
#                             📊 Vector Search
#                         </span>
#                         <span style="background: rgba(255,255,255,0.2); padding: 0.5rem 1rem; border-radius: 20px; margin: 0 0.5rem;">
#                             🕸️ Knowledge Graph
#                         </span>
#                     </div>
#                 </div>
#                 """)
            
#             # Status indicator
#             with gr.Row():
#                 status_indicator = gr.HTML(
#                     value=f"""
#                     <div style="text-align: center; padding: 1rem;">
#                         <span class="{'status-online' if self.setup_complete else 'status-loading'}">
#                             {'🟢 System Ready - Ask me anything!' if self.setup_complete else '🟡 Initializing AI systems...'}
#                         </span>
#                     </div>
#                     """ 
#                 )
            
#             # Main interface layout
#             with gr.Row():
#                 # Left column - Chat interface
#                 with gr.Column(scale=3, elem_classes="chat-container"):
#                     chatbot = gr.Chatbot(
#                         label="💬 AI Business Consultant",
#                         height=600,
#                         show_label=True,
#                         container=True,
#                         show_copy_button=True,
#                         bubble_full_width=False,
#                         layout="panel",
#                         elem_classes="message"
#                     )
                    
#                     with gr.Row():
#                         msg = gr.Textbox(
#                             label="",
#                             placeholder="💼 Ask me about business strategies, case studies, or request analysis...",
#                             container=False,
#                             scale=4,
#                             lines=2
#                         )
#                         with gr.Column(scale=1, min_width=120):
#                             submit_btn = gr.Button("🚀 Analyze", variant="primary", elem_classes="primary-btn")
#                             clear_btn = gr.Button("🗑️ Clear", elem_classes="secondary-btn")
                
#                 # Right column - Enhanced sidebar
#                 with gr.Column(scale=1, elem_classes="sidebar"):
#                     gr.HTML("<h3 style='text-align: center; color: #667eea;'>📊 Control Center</h3>")
                    
#                     # Session stats with enhanced display
#                     stats_display = gr.Markdown(
#                         value=self.get_session_stats(),
#                         elem_classes="stats-card"
#                     )
                    
#                     refresh_stats_btn = gr.Button("🔄 Refresh Stats", size="sm")
                    
#                     # System status
#                     system_status = gr.Markdown(
#                         value=self.get_system_status(),
#                         elem_classes="stats-card"
#                     )
                    
#                     # Quick tips
#                     gr.HTML("""
#                     <div style="margin-top: 2rem;">
#                         <h4 style="color: #667eea;">💡 Pro Tips</h4>
#                         <ul style="font-size: 0.9rem; line-height: 1.6;">
#                             <li>📝 Be specific for better insights</li>
#                             <li>🔍 Request analysis for deep dives</li>
#                             <li>📊 Ask for metrics and KPIs</li>
#                             <li>🎯 Sources are auto-cited</li>
#                             <li>🚀 Try different query types</li>
#                         </ul>
#                     </div>
#                     """)
            
#             # Enhanced examples section with cards
#             gr.HTML("<h3 style='text-align: center; margin: 2rem 0 1rem 0; color: #667eea;'>🎯 Try These Smart Queries</h3>")
            
#             with gr.Row():
#                 example_buttons = []
#                 for i in range(0, len(self.example_queries), 2):
#                     with gr.Column():
#                         for j in range(2):
#                             if i + j < len(self.example_queries):
#                                 emoji, query = self.example_queries[i + j]
#                                 btn = gr.Button(
#                                     value=f"{emoji}\n{query[:50]}{'...' if len(query) > 50 else ''}",
#                                     elem_classes="example-card",
#                                     size="lg"
#                                 )
#                                 example_buttons.append((btn, query))
            
#             # Event handlers
#             msg.submit(
#                 self.chat_interface,
#                 inputs=[msg, chatbot],
#                 outputs=[msg, chatbot]
#             )
            
#             submit_btn.click(
#                 self.chat_interface,
#                 inputs=[msg, chatbot],
#                 outputs=[msg, chatbot]
#             )
            
#             clear_btn.click(
#                 self.clear_conversation,
#                 outputs=[chatbot]
#             )
            
#             refresh_stats_btn.click(
#                 self.get_session_stats,
#                 outputs=[stats_display]
#             )
            
#             # Connect example buttons
#             for btn, query in example_buttons:
#                 btn.click(
#                     lambda q=query: q,
#                     outputs=[msg]
#                 )
            
#             # Footer with credits and info
#             gr.HTML("""
#             <div style="text-align: center; margin-top: 3rem; padding: 2rem; background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%); border-radius: 15px;">
#                 <h4 style="color: #667eea; margin-bottom: 1rem;">🚀 Powered by Advanced AI</h4>
#                 <div style="display: flex; justify-content: center; flex-wrap: wrap; gap: 2rem; margin-bottom: 1rem;">
#                     <div style="text-align: center;">
#                         <div style="font-size: 2rem;">🧠</div>
#                         <strong>LLM Analysis</strong><br>
#                         <small>GPT/Gemini powered insights</small>
#                     </div>
#                     <div style="text-align: center;">
#                         <div style="font-size: 2rem;">🔍</div>
#                         <strong>Vector Search</strong><br>
#                         <small>FAISS semantic matching</small>
#                     </div>
#                     <div style="text-align: center;">
#                         <div style="font-size: 2rem;">🕸️</div>
#                         <strong>Knowledge Graph</strong><br>
#                         <small>NetworkX relationships</small>
#                     </div>
#                     <div style="text-align: center;">
#                         <div style="font-size: 2rem;">📊</div>
#                         <strong>RAG Architecture</strong><br>
#                         <small>Retrieval augmented generation</small>
#                     </div>
#                 </div>
#                 <p style="color: #666; font-size: 0.9rem; margin: 0;">
#                     Built with ❤️ using Gradio • sentence-transformers • NetworkX • Advanced Prompt Engineering
#                 </p>
#             </div>
#             """)
        
#         return interface
    
#     def _get_llm_provider(self) -> str:
#         """Get current LLM provider name with emoji."""
#         from src.config import LLM_PROVIDER
#         provider_map = {
#             'gemini': '🔮 Gemini',
#             'openai': '🤖 OpenAI',
#             'anthropic': '🧠 Claude'
#         }
#         return provider_map.get(LLM_PROVIDER.lower(), f'🤖 {LLM_PROVIDER.title()}')
    
#     def launch(self, **kwargs) -> None:
#         """Launch the enhanced Gradio interface."""
#         interface = self.create_interface()
        
#         launch_kwargs = {
#             'server_port': GRADIO_PORT,
#             'share': GRADIO_SHARE,
#             'show_error': True,
#             'quiet': False,
#             'favicon_path': None,  # Could add custom favicon
#             'ssl_verify': False,
#             **kwargs
#         }
        
#         print(f"""
# 🚀 Launching Enhanced AI Business Case Study Analyzer
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# 🌐 URL: http://localhost:{GRADIO_PORT}
# 🧠 LLM: {self._get_llm_provider()}
# 📊 Status: {'✅ Ready' if self.setup_complete else '⚠️ Initializing'}
# 🔍 Features: Vector Search + Knowledge Graph + RAG

# 💡 Try asking:
#    • "What are effective customer retention strategies?"
#    • "Analyze digital transformation success factors"
#    • "Compare cost reduction approaches"

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#         """)
        
#         logger.info(f"🚀 Launching enhanced Gradio interface on port {GRADIO_PORT}")
#         interface.launch(**launch_kwargs)


# class QuickStart:
#     """Quick start helper for easy setup."""
    
#     @staticmethod
#     def setup_environment():
#         """Quick environment setup."""
#         print("🔧 Setting up environment...")
        
#         # Check for API keys
#         from src.config import GEMINI_API_KEY, OPENAI_API_KEY, LLM_PROVIDER
        
#         if not GEMINI_API_KEY and not OPENAI_API_KEY:
#             print("""
# ⚠️  API Key Required
# ━━━━━━━━━━━━━━━━━━━━

# Please set up your API key in the .env file:

# For Gemini (Free tier available):
# GEMINI_API_KEY=your_key_here

# For OpenAI:
# OPENAI_API_KEY=your_key_here
# LLM_PROVIDER=openai

# Get your keys:
# 🔮 Gemini: https://makersuite.google.com/app/apikey
# 🤖 OpenAI: https://platform.openai.com/api-keys
#             """)
#             return False
        
#         print(f"✅ API configured: {LLM_PROVIDER.title()}")
        
#         # Check for sample data
#         if not (Path("data") / "cases.docx").exists():
#             print("📝 Creating sample data...")
#             try:
#                 from create_sample_data import main as create_data
#                 create_data()
#                 print("✅ Sample data created")
#             except Exception as e:
#                 print(f"⚠️ Could not create sample data: {e}")
        
#         return True
    
#     @staticmethod
#     def run():
#         """Quick run with setup check."""
#         if QuickStart.setup_environment():
#             ui = EnhancedChatbotUI()
#             ui.launch()
#         else:
#             print("❌ Setup incomplete. Please configure API keys.")


# def main():
#     """Main entry point for the enhanced UI."""
#     import sys
#     import argparse
    
#     parser = argparse.ArgumentParser(description="AI Business Case Study Analyzer")
#     parser.add_argument("--port", type=int, default=GRADIO_PORT, help="Port to run on")
#     parser.add_argument("--share", action="store_true", help="Create public link")
#     parser.add_argument("--quick", action="store_true", help="Quick start with setup")
    
#     args = parser.parse_args()
    
#     # Configure enhanced logging
#     logging.basicConfig(
#         level=logging.INFO,
#         format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
#         handlers=[
#             logging.StreamHandler(sys.stdout),
#         ]
#     )
    
#     try:
#         if args.quick:
#             QuickStart.run()
#         else:
#             # Create and launch enhanced UI
#             ui = EnhancedChatbotUI()
#             ui.launch(server_port=args.port, share=args.share)
            
#     except KeyboardInterrupt:
#         print("\n👋 Shutting down AI Business Case Study Analyzer")
#     except Exception as e:
#         logger.error(f"❌ Error running enhanced UI: {e}")
#         print(f"""
# 🚨 Startup Error
# ━━━━━━━━━━━━━━━

# Error: {e}

# 💡 Troubleshooting:
# 1. Check your .env file has valid API keys
# 2. Run: pip install -r requirements.txt
# 3. Try: python -m src.ui --quick

# Need help? Check the README.md file.
#         """)
#         sys.exit(1)


# if __name__ == "__main__":
#     main()

"""
Simple, creative UI for arbitration and legal document chatbot.
Focused on clean design without unnecessary complexity.
"""
from typing import Dict, Any, List, Tuple, Optional
import gradio as gr
import logging
import time
from pathlib import Path

from src.config import get_logger, GRADIO_PORT, GRADIO_SHARE, DATA_ROOT
from src.chatbot import handle_message, get_session_info, clear_user_session
from src.embeddings import build_and_save_embeddings, load_index
from src.knowledge_graph import load_kg, build_kg, save_kg

logger = get_logger(__name__)


class SimpleChatbotUI:
    """Simple, creative UI focused on legal and arbitration document analysis."""
    
    def __init__(self):
        """Initialize the simple UI."""
        self.current_user = "user"
        self.setup_complete = False
        
        # Simple example queries focused on your document types
        self.example_queries = [
            ("⚖️ Arbitration Process", "What is the timeline for arbitration proceedings?"),
            ("📋 Cost Analysis", "Compare arbitration costs to claimed amounts"),
            ("🏛️ Legal Precedents", "Find relevant arbitration case precedents"),
            ("📊 Decision Trees", "Explain the arbitration decision process"),
            ("⏰ Emergency Procedures", "What are emergency arbitration conditions?"),
            ("✅ Clause Validity", "How to validate an arbitration clause?"),
        ]
        
        # Initialize system
        self._initialize_system()
    
    def _initialize_system(self) -> None:
        """Initialize the document processing system."""
        try:
            logger.info("🚀 Initializing document analysis system...")
            
            # Try to load existing indexes
            embeddings_loaded = load_index()
            
            try:
                kg_loaded = load_kg()
            except Exception as e:
                logger.warning(f"Knowledge graph loading failed: {e}")
                kg_loaded = False
            
            if not embeddings_loaded or not kg_loaded:
                logger.info("📚 Building document indexes...")
                self._build_indexes()
            
            self.setup_complete = True
            logger.info("✅ System ready for document analysis")
            
        except Exception as e:
            logger.error(f"❌ System initialization failed: {e}")
            self.setup_complete = False
    
    def _build_indexes(self) -> None:
        """Build document indexes from available files."""
        # Check for documents in data directory
        data_paths = [
            DATA_ROOT / "*.docx",
            DATA_ROOT / "batch_1" / "*.docx"
        ]
        
        # Find any .docx files
        import glob
        doc_files = []
        for pattern in data_paths:
            doc_files.extend(glob.glob(str(pattern)))
        
        if doc_files:
            # Use first available document
            build_and_save_embeddings(Path(doc_files[0]))
        else:
            # Build with sample data
            build_and_save_embeddings()
        
        # Build knowledge graph
        try:
            kg = build_kg()
            save_kg(kg)
        except Exception as e:
            logger.warning(f"Knowledge graph building failed: {e}")
    
    def chat_interface(self, message: str, history: List[List[str]]) -> Tuple[str, List[List[str]]]:
        """Handle chat interactions with simple formatting."""
        if not self.setup_complete:
            return "", history + [[message, "⏳ System is initializing... Please wait a moment."]]
        
        if not message.strip():
            return "", history
        
        try:
            # Get response from chatbot
            response = handle_message(self.current_user, message.strip())
            
            # Simple response formatting
            formatted_response = self._format_simple_response(response)
            
            # Update history
            new_history = history + [[message, formatted_response]]
            
            return "", new_history
            
        except Exception as e:
            logger.error(f"Chat error: {e}")
            error_response = f"❌ Error processing your question: {str(e)}"
            return "", history + [[message, error_response]]
    
    def _format_simple_response(self, response: Dict) -> str:
        """Format response simply and cleanly."""
        answer = response['answer']
        sources = response['sources']
        confidence = response['confidence']
        query_type = response.get('query_type', 'general')
        
        # Main answer
        formatted = answer
        
        # Simple footer with key info
        formatted += f"\n\n---\n"
        
        # Confidence indicator
        if confidence >= 0.8:
            formatted += f"🎯 **High Confidence** ({confidence:.0%}) - Strong document support\n"
        elif confidence >= 0.6:
            formatted += f"✅ **Good Confidence** ({confidence:.0%}) - Adequate document evidence\n"
        else:
            formatted += f"⚠️ **Moderate Confidence** ({confidence:.0%}) - Limited document support\n"
        
        # Source count
        if sources:
            formatted += f"📚 **Sources:** {len(sources)} documents analyzed\n"
        
        # Query type
        query_icons = {
            'analysis': '📊',
            'recommendation': '💡',
            'direct_answer': '🎯',
            'comparison': '⚖️'
        }
        icon = query_icons.get(query_type, '📝')
        formatted += f"{icon} **Type:** {query_type.title().replace('_', ' ')}"
        
        return formatted
    
    def clear_conversation(self) -> List[List[str]]:
        """Clear the conversation."""
        clear_user_session(self.current_user)
        logger.info(f"🗑️ Cleared conversation")
        return []
    
    def get_session_stats(self) -> str:
        """Get simple session statistics."""
        session_info = get_session_info(self.current_user)
        
        if session_info:
            total_turns = session_info['total_turns']
            avg_confidence = session_info.get('avg_confidence', 0)
            
            stats = f"## 📊 Session Info\n\n"
            stats += f"**Questions Asked:** {total_turns}\n"
            
            if avg_confidence > 0:
                confidence_level = "High" if avg_confidence >= 0.7 else "Good" if avg_confidence >= 0.5 else "Fair"
                stats += f"**Answer Quality:** {confidence_level}\n"
            
            stats += f"**Focus:** Legal & Arbitration Documents"
        else:
            stats = "## 📊 Session Info\n\nNew session - ask your first question!"
        
        return stats
    
    def create_interface(self) -> gr.Blocks:
        """Create the simple, creative interface."""
        
        # Clean, modern CSS
        simple_css = """
        .gradio-container {
            max-width: 1200px !important;
            margin: 0 auto;
            font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, sans-serif;
        }
        
        .main-header {
            background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
            color: white;
            padding: 2rem;
            border-radius: 15px;
            margin-bottom: 2rem;
            text-align: center;
            box-shadow: 0 4px 20px rgba(30, 60, 114, 0.3);
        }
        
        .chat-container {
            background: linear-gradient(145deg, #f8f9fa, #e9ecef);
            border-radius: 15px;
            padding: 1rem;
            min-height: 500px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        
        .sidebar {
            background: rgba(255, 255, 255, 0.95);
            border-radius: 15px;
            padding: 1.5rem;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        
        .example-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border-radius: 12px;
            padding: 1rem;
            margin: 0.5rem;
            cursor: pointer;
            transition: all 0.3s ease;
            border: none;
            text-align: left;
        }
        
        .example-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
        }
        
        .send-btn {
            background: linear-gradient(45deg, #667eea, #764ba2);
            color: white;
            border: none;
            border-radius: 20px;
            padding: 0.8rem 2rem;
            font-weight: 600;
            transition: all 0.3s ease;
        }
        
        .send-btn:hover {
            transform: translateY(-1px);
            box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);
        }
        
        .clear-btn {
            background: #6c757d;
            color: white;
            border: none;
            border-radius: 20px;
            padding: 0.8rem 1.5rem;
            transition: all 0.3s ease;
        }
        
        .clear-btn:hover {
            background: #5a6268;
            transform: translateY(-1px);
        }
        """
        
        with gr.Blocks(
            theme=gr.themes.Soft(
                primary_hue="blue",
                secondary_hue="purple",
                radius_size="lg"
            ),
            css=simple_css,
            title="⚖️ Legal Document AI Assistant"
        ) as interface:
            
            # Simple header
            with gr.Row(elem_classes="main-header"):
                gr.HTML("""
                <div>
                    <h1 style="font-size: 2.5rem; margin-bottom: 1rem;">
                        ⚖️ Legal Document AI Assistant
                    </h1>
                    <p style="font-size: 1.1rem; opacity: 0.9;">
                        AI-powered analysis of arbitration cases, legal procedures, and decision trees
                    </p>
                </div>
                """)
            
            # Status indicator
            with gr.Row():
                status_html = f"""
                <div style="text-align: center; padding: 1rem;">
                    <span style="color: {'#28a745' if self.setup_complete else '#ffc107'}; font-weight: bold;">
                        {'🟢 Ready for Questions' if self.setup_complete else '🟡 Initializing System...'}
                    </span>
                </div>
                """
                gr.HTML(status_html)
            
            # Main layout
            with gr.Row():
                # Chat area
                with gr.Column(scale=3, elem_classes="chat-container"):
                    chatbot = gr.Chatbot(
                        label="💬 Legal Document Analysis",
                        height=500,
                        show_copy_button=True,
                        bubble_full_width=False
                    )
                    
                    with gr.Row():
                        msg = gr.Textbox(
                            label="",
                            placeholder="💼 Ask about arbitration procedures, legal costs, decision trees, or case precedents...",
                            container=False,
                            scale=4,
                            lines=2
                        )
                        with gr.Column(scale=1, min_width=100):
                            send_btn = gr.Button("📤 Send", elem_classes="send-btn")
                            clear_btn = gr.Button("🗑️ Clear", elem_classes="clear-btn")
                
                # Simple sidebar
                with gr.Column(scale=1, elem_classes="sidebar"):
                    gr.HTML("<h3 style='text-align: center; color: #667eea;'>📊 Session</h3>")
                    
                    session_stats = gr.Markdown(
                        value=self.get_session_stats()
                    )
                    
                    refresh_btn = gr.Button("🔄 Refresh", size="sm")
                    
                    gr.HTML("""
                    <div style="margin-top: 2rem;">
                        <h4 style="color: #667eea;">💡 Tips</h4>
                        <ul style="font-size: 0.9rem; line-height: 1.6;">
                            <li>📋 Ask about arbitration timelines</li>
                            <li>💰 Compare costs and claims</li>
                            <li>🏛️ Find legal precedents</li>
                            <li>📊 Analyze decision trees</li>
                            <li>⚡ Emergency procedures</li>
                        </ul>
                    </div>
                    """)
            
            # Example questions
            gr.HTML("<h3 style='text-align: center; margin: 2rem 0 1rem 0; color: #667eea;'>🎯 Quick Questions</h3>")
            
            with gr.Row():
                example_buttons = []
                for i in range(0, len(self.example_queries), 2):
                    with gr.Column():
                        for j in range(2):
                            if i + j < len(self.example_queries):
                                emoji, query = self.example_queries[i + j]
                                btn = gr.Button(
                                    value=f"{emoji}\n{query}",
                                    elem_classes="example-card",
                                    size="lg"
                                )
                                example_buttons.append((btn, query))
            
            # Event handlers
            msg.submit(
                self.chat_interface,
                inputs=[msg, chatbot],
                outputs=[msg, chatbot]
            )
            
            send_btn.click(
                self.chat_interface,
                inputs=[msg, chatbot],
                outputs=[msg, chatbot]
            )
            
            clear_btn.click(
                self.clear_conversation,
                outputs=[chatbot]
            )
            
            refresh_btn.click(
                self.get_session_stats,
                outputs=[session_stats]
            )
            
            # Connect example buttons
            for btn, query in example_buttons:
                btn.click(
                    lambda q=query: q,
                    outputs=[msg]
                )
            
            # Simple footer
            gr.HTML("""
            <div style="text-align: center; margin-top: 2rem; padding: 1.5rem; background: #f8f9fa; border-radius: 10px;">
                <p style="color: #666; margin: 0;">
                    ⚖️ <strong>Legal Document AI Assistant</strong> • Powered by Advanced NLP • 
                    Specialized in Arbitration & Legal Analysis
                </p>
            </div>
            """)
        
        return interface
    
    def launch(self, **kwargs) -> None:
        """Launch the simple interface."""
        interface = self.create_interface()
        
        launch_kwargs = {
            'server_port': GRADIO_PORT,
            'share': GRADIO_SHARE,
            'show_error': True,
            'quiet': False,
            **kwargs
        }
        
        print(f"""
⚖️ Legal Document AI Assistant
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🌐 URL: http://localhost:{GRADIO_PORT}
📚 Focus: Arbitration & Legal Documents
🎯 Status: {'✅ Ready' if self.setup_complete else '⚠️ Initializing'}

💡 Ask about:
   • Arbitration procedures and timelines
   • Cost analysis and comparisons
   • Legal precedents and case studies
   • Decision trees and emergency conditions

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        """)
        
        logger.info(f"🚀 Launching Legal Document AI Assistant on port {GRADIO_PORT}")
        interface.launch(**launch_kwargs)


def main():
    """Main entry point for the simple UI."""
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(description="Legal Document AI Assistant")
    parser.add_argument("--port", type=int, default=GRADIO_PORT, help="Port to run on")
    parser.add_argument("--share", action="store_true", help="Create public link")
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    
    try:
        # Create and launch simple UI
        ui = SimpleChatbotUI()
        ui.launch(server_port=args.port, share=args.share)
        
    except KeyboardInterrupt:
        print("\n👋 Legal Document AI Assistant stopped")
    except Exception as e:
        logger.error(f"❌ Error running UI: {e}")
        print(f"""
🚨 Startup Error
━━━━━━━━━━━━━━━

Error: {e}

💡 Try:
1. Check your .env file has valid API keys
2. Run: pip install -r requirements.txt
3. Ensure documents are in data/ folder

Need help? Check the logs above.
        """)
        sys.exit(1)


if __name__ == "__main__":
    main()