
### src/ui.py
"""
Gradio web interface for the case study chatbot.
"""
from typing import Dict, Any, List, Tuple, Optional
import gradio as gr
import logging
import json
from pathlib import Path

from .config import get_logger, GRADIO_PORT, GRADIO_SHARE, DATA_ROOT
from .chatbot import handle_message, get_session_info, clear_user_session
from .embeddings import build_and_save_embeddings, load_index
from .knowledge_graph import load_kg, build_kg, save_kg
from .data_loader import load_docx

logger = get_logger(__name__)


class ChatbotUI:
    """Gradio UI for the case study chatbot."""
    
    def __init__(self):
        """Initialize the UI."""
        self.current_user = "default_user"
        self.setup_complete = False
        
        # Initialize system
        self._initialize_system()
    
    def _initialize_system(self) -> None:
        """Initialize embeddings and knowledge graph."""
        try:
            logger.info("Initializing chatbot system...")
            
            # Try to load existing indexes
            embeddings_loaded = load_index()
            kg_loaded = load_kg()
            
            if not embeddings_loaded or not kg_loaded:
                logger.info("Building new indexes from documents...")
                self._build_indexes()
            
            self.setup_complete = True
            logger.info("System initialization complete")
            
        except Exception as e:
            logger.error(f"System initialization failed: {e}")
            self.setup_complete = False
    
    def _build_indexes(self) -> None:
        """Build embeddings and knowledge graph from documents."""
        doc_path = DATA_ROOT / "cases.docx"
        
        if doc_path.exists():
            # Load and process document
            items = load_docx(doc_path)
            logger.info(f"Loaded {len(items)} items from document")
            
            # Build embeddings
            build_and_save_embeddings(doc_path)
            
            # Build knowledge graph
            from .knowledge_graph import build_kg
            build_kg(items)
            save_kg()
            
        else:
            # Build with sample data
            logger.warning("No document found, building with sample data")
            build_and_save_embeddings()
    
    def chat_interface(self, message: str, history: List[List[str]]) -> Tuple[str, List[List[str]]]:
        """
        Handle chat interface interactions.
        
        Args:
            message: User message
            history: Chat history
            
        Returns:
            Tuple of (empty string, updated history)
        """
        if not self.setup_complete:
            return "", history + [[message, "⚠️ System is still initializing. Please wait..."]]
        
        if not message.strip():
            return "", history
        
        try:
            # Get response from chatbot
            response = handle_message(self.current_user, message.strip())
            
            # Format response with sources
            formatted_response = self._format_response(response)
            
            # Update history
            new_history = history + [[message, formatted_response]]
            
            return "", new_history
            
        except Exception as e:
            logger.error(f"Chat interface error: {e}")
            error_response = f"❌ Sorry, I encountered an error: {str(e)}"
            return "", history + [[message, error_response]]
    
    def _format_response(self, response: Dict) -> str:
        """Format the chatbot response for display."""
        answer = response['answer']
        sources = response['sources']
        confidence = response['confidence']
        processing_time = response['processing_time']
        
        # Main answer
        formatted = answer
        
        # Add confidence and timing info
        formatted += f"\n\n---\n"
        formatted += f"**Confidence:** {confidence:.1%} | **Processing Time:** {processing_time:.2f}s"
        
        # Add sources if available
        if sources:
            formatted += f"\n\n**Sources:**"
            for i, source in enumerate(sources[:3], 1):  # Show top 3 sources
                source_type = source.get('type', 'document')
                source_text = source['text'][:150] + "..." if len(source['text']) > 150 else source['text']
                formatted += f"\n{i}. [{source_type.upper()}] {source_text}"
        
        # Add KG info if available
        kg_nodes = response.get('kg_nodes', [])
        if kg_nodes:
            formatted += f"\n\n**Related Knowledge:** {len(kg_nodes)} connections found"
        
        return formatted
    
    def clear_conversation(self) -> List[List[str]]:
        """Clear the current conversation."""
        clear_user_session(self.current_user)
        logger.info(f"Cleared conversation for user {self.current_user}")
        return []
    
    def get_session_stats(self) -> str:
        """Get current session statistics."""
        session_info = get_session_info(self.current_user)
        
        if session_info:
            stats = f"""**Session Statistics:**
- Total turns: {session_info['total_turns']}
- Average confidence: {session_info.get('avg_confidence', 0):.1%}
- Average response time: {session_info.get('avg_processing_time', 0):.2f}s
- Query types: {', '.join(session_info.get('query_types', []))}"""
        else:
            stats = "No active session"
        
        return stats
    
    def create_interface(self) -> gr.Blocks:
        """Create the Gradio interface."""
        
        # Custom CSS for better styling
        custom_css = """
        .gradio-container {
            max-width: 1200px !important;
        }
        
        .chat-container {
            height: 500px;
            overflow-y: auto;
        }
        
        .source-box {
            background-color: #f0f0f0;
            padding: 10px;
            border-radius: 5px;
            margin: 5px 0;
        }
        """
        
        with gr.Blocks(
            theme=gr.themes.Soft(),
            css=custom_css,
            title="Case Study Chatbot"
        ) as interface:
            
            # Header
            gr.Markdown(
                """
                # 📊 Case Study Conversational Chatbot
                
                Welcome to the AI-powered case study analysis assistant. Ask questions about business cases, 
                request analysis, or explore insights from the knowledge base.
                
                **Example queries:**
                - "What are the main customer service problems?"
                - "How did companies solve their turnover issues?"
                - "Analyze the effectiveness of training programs"
                """
            )
            
            # Status indicator
            with gr.Row():
                status_text = "🟢 System Ready" if self.setup_complete else "🟡 Initializing..."
                gr.Markdown(f"**Status:** {status_text}")
            
            # Main chat interface
            with gr.Row():
                with gr.Column(scale=3):
                    chatbot = gr.Chatbot(
                        label="Conversation",
                        height=500,
                        show_label=True,
                        container=True,
                        show_copy_button=True
                    )
                    
                    msg = gr.Textbox(
                        label="Your message",
                        placeholder="Ask me about case studies...",
                        container=True,
                        scale=4
                    )
                    
                    with gr.Row():
                        submit_btn = gr.Button("Send", variant="primary", scale=1)
                        clear_btn = gr.Button("Clear Chat", scale=1)
                
                # Sidebar with info
                with gr.Column(scale=1):
                    gr.Markdown("### 📈 Session Info")
                    
                    stats_display = gr.Markdown("No active session")
                    
                    refresh_stats_btn = gr.Button("Refresh Stats", size="sm")
                    
                    gr.Markdown("### 💡 Tips")
                    gr.Markdown(
                        """
                        - Ask specific questions for better results
                        - Request analysis for deeper insights
                        - Sources are automatically cited
                        - Use "Clear Chat" to start over
                        """
                    )
                    
                    # System info
                    gr.Markdown("### ⚙️ System Info")
                    system_info = f"""
                    - **LLM Provider:** {self._get_llm_provider()}
                    - **Embedding Model:** all-MiniLM-L6-v2
                    - **Vector Index:** FAISS
                    - **Knowledge Graph:** NetworkX
                    """
                    gr.Markdown(system_info)
            
            # Event handlers
            msg.submit(
                self.chat_interface,
                inputs=[msg, chatbot],
                outputs=[msg, chatbot]
            )
            
            submit_btn.click(
                self.chat_interface,
                inputs=[msg, chatbot],
                outputs=[msg, chatbot]
            )
            
            clear_btn.click(
                self.clear_conversation,
                outputs=[chatbot]
            )
            
            refresh_stats_btn.click(
                self.get_session_stats,
                outputs=[stats_display]
            )
            
            # Examples
            gr.Examples(
                examples=[
                    "What are the main problems companies face?",
                    "How can businesses improve customer satisfaction?",
                    "Analyze the effectiveness of employee training programs",
                    "What solutions work best for reducing costs?",
                    "Find examples of successful problem-solving approaches"
                ],
                inputs=msg
            )
            
            # Footer
            gr.Markdown(
                """
                ---
                **Case Study Chatbot PoC** | Built with Gradio, FAISS, NetworkX, and LLM APIs
                """
            )
        
        return interface
    
    def _get_llm_provider(self) -> str:
        """Get current LLM provider name."""
        from .config import LLM_PROVIDER
        return LLM_PROVIDER.title()
    
    def launch(self, **kwargs) -> None:
        """Launch the Gradio interface."""
        interface = self.create_interface()
        
        launch_kwargs = {
            'server_port': GRADIO_PORT,
            'share': GRADIO_SHARE,
            'show_error': True,
            'quiet': False,
            **kwargs
        }
        
        logger.info(f"Launching Gradio interface on port {GRADIO_PORT}")
        interface.launch(**launch_kwargs)


def main():
    """Main entry point for the UI."""
    import sys
    
    # Configure logging for UI
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
        ]
    )
    
    # Create and launch UI
    ui = ChatbotUI()
    
    try:
        ui.launch()
    except KeyboardInterrupt:
        logger.info("Shutting down chatbot UI")
    except Exception as e:
        logger.error(f"Error running UI: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()