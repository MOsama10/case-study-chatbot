
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
