#!/usr/bin/env python3
"""
Main launcher for Legal Document AI Assistant.
Place this file in your project root directory (same level as src/ folder).
"""

import sys
import os
import subprocess
import logging
from pathlib import Path
import time

def setup_logging():
    """Setup basic logging."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )
    return logging.getLogger(__name__)

logger = setup_logging()

class QuickSetup:
    """Quick setup and launch for the legal document assistant."""
    
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.src_dir = self.project_root / "src"
        self.data_dir = self.project_root / "data"
        
        # Add src to Python path
        if str(self.src_dir) not in sys.path:
            sys.path.insert(0, str(self.src_dir))
    
    def check_python_version(self) -> bool:
        """Check Python version."""
        version = sys.version_info
        if version.major < 3 or (version.major == 3 and version.minor < 10):
            logger.error(f"Python 3.10+ required, found {version.major}.{version.minor}")
            return False
        logger.info(f"✅ Python {version.major}.{version.minor} OK")
        return True
    
    def check_structure(self) -> bool:
        """Check project structure."""
        if not self.src_dir.exists():
            logger.error(f"src/ directory not found in {self.project_root}")
            return False
        
        required_files = [
            self.src_dir / "config.py",
            self.src_dir / "agent.py", 
            self.src_dir / "ui.py"
        ]
        
        for file_path in required_files:
            if not file_path.exists():
                logger.error(f"Required file missing: {file_path}")
                return False
        
        logger.info("✅ Project structure OK")
        return True
    
    def install_packages(self) -> bool:
        """Install essential packages."""
        logger.info("📦 Installing essential packages...")
        
        essential = [
            "python-dotenv>=1.0.0",
            "gradio>=3.50.0", 
            "numpy>=1.24.0",
            "networkx>=3.1.0"
        ]
        
        try:
            for package in essential:
                package_name = package.split(">=")[0]
                try:
                    # Try to import to check if installed
                    if package_name == "python-dotenv":
                        import dotenv
                    elif package_name == "networkx":
                        import networkx
                    else:
                        __import__(package_name.replace("-", "_"))
                    logger.info(f"✅ {package_name} already available")
                except ImportError:
                    logger.info(f"📦 Installing {package_name}...")
                    subprocess.check_call([
                        sys.executable, "-m", "pip", "install", package, "--quiet"
                    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            
            # Try optional ML packages
            optional = ["sentence-transformers>=2.2.2", "faiss-cpu>=1.7.4"]
            for package in optional:
                package_name = package.split(">=")[0]
                try:
                    subprocess.check_call([
                        sys.executable, "-m", "pip", "install", package, "--quiet"
                    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    logger.info(f"✅ {package_name} installed")
                except:
                    logger.warning(f"⚠️ {package_name} installation failed (will use fallback)")
            
            return True
        except Exception as e:
            logger.warning(f"Package installation issues: {e}")
            return True  # Continue anyway
    
    def setup_env(self) -> bool:
        """Setup environment file."""
        env_file = self.project_root / ".env"
        
        if not env_file.exists():
            logger.info("📝 Creating .env file...")
            env_content = """# Legal Document AI Assistant Configuration

# Choose your AI provider:
# Option 1: Gemini (Free tier) - Get key at: https://makersuite.google.com/app/apikey
GEMINI_API_KEY=your_gemini_api_key_here
LLM_PROVIDER=gemini

# Option 2: OpenAI (Paid) - Get key at: https://platform.openai.com/api-keys  
# OPENAI_API_KEY=your_openai_api_key_here
# LLM_PROVIDER=openai

# System settings
GRADIO_PORT=7860
LLM_TEMPERATURE=0.2
LLM_MAX_TOKENS=2000
"""
            env_file.write_text(env_content)
            logger.info("✅ Created .env file")
            logger.info("🔑 IMPORTANT: Edit .env and add your API key!")
            return False
        
        # Check API key
        try:
            from dotenv import load_dotenv
            load_dotenv()
            
            gemini_key = os.getenv("GEMINI_API_KEY")
            openai_key = os.getenv("OPENAI_API_KEY")
            
            if (not gemini_key or gemini_key == "your_gemini_api_key_here") and \
               (not openai_key or openai_key == "your_openai_api_key_here"):
                logger.warning("⚠️ API key not configured!")
                logger.info("📝 Edit .env file and add your API key, then run again")
                return False
            
            logger.info("✅ API key configured")
            return True
        except Exception as e:
            logger.error(f"Environment check failed: {e}")
            return False
    
    def create_sample_data(self) -> None:
        """Create sample data if needed."""
        self.data_dir.mkdir(exist_ok=True)
        
        # Check for existing data
        existing_files = (
            list(self.data_dir.glob("*.docx")) + 
            list(self.data_dir.glob("*.txt")) +
            list((self.data_dir / "batch_1").glob("*.docx"))
        )
        
        if existing_files:
            logger.info(f"✅ Found {len(existing_files)} data files")
            return
        
        logger.info("📝 Creating sample legal data...")
        sample_content = """LEGAL DOCUMENT AI ASSISTANT - SAMPLE DATA

=== Arbitration Timeline ===
Standard arbitration: 6-18 months from initiation to final award
Emergency arbitration: 14-30 days for urgent matters

=== Arbitration Costs ===
Typical range: $50,000 to $500,000 depending on complexity
- Administrative fees: 5-15% of total costs
- Arbitrator fees: 60-70% of total costs
- Legal representation: 20-30% of total costs

=== Clause Validity ===
Valid arbitration clauses require:
1. Clear and unambiguous language
2. Mutual agreement by all parties  
3. Proper scope definition
4. Designated arbitration rules and institution

=== Emergency Conditions ===
Emergency arbitration available when:
- Immediate harm would occur without relief
- Regular arbitration would be too slow
- Interim measures necessary to preserve status quo

=== Award Enforcement ===
Arbitral awards are generally final and binding
Limited grounds for challenge:
- Procedural irregularities
- Arbitrator misconduct
- Award exceeds scope of agreement
"""
        
        sample_file = self.data_dir / "sample_legal_data.txt"
        sample_file.write_text(sample_content)
        logger.info(f"✅ Created sample data: {sample_file}")
    
    def test_imports(self) -> bool:
        """Test critical imports."""
        logger.info("🧪 Testing system imports...")
        
        try:
            # Test imports one by one for better error reporting
            logger.info("   Testing config...")
            import src.config
            
            logger.info("   Testing data loader...")
            import src.data_loader
            
            logger.info("   Testing embeddings...")
            import src.embeddings
            
            logger.info("   Testing knowledge graph...")
            import src.knowledge_graph
            
            logger.info("   Testing agent...")
            import src.agent
            
            logger.info("   Testing chatbot...")
            import src.chatbot
            # Test specific function
            from src.chatbot import handle_message
            
            logger.info("   Testing UI...")
            import src.ui
            from src.ui import SimpleChatbotUI
            
            logger.info("✅ All imports successful")
            return True
        except ImportError as e:
            logger.error(f"❌ Import failed: {e}")
            logger.error("💡 Try running the quick_fix.py script first")
            return False
        except Exception as e:
            logger.error(f"❌ Import error: {e}")
            return False
    
    def launch_app(self) -> bool:
        """Launch the application."""
        try:
            logger.info("🚀 Launching Legal Document AI Assistant...")
            
            from src.ui import SimpleChatbotUI
            chatbot_ui = SimpleChatbotUI()
            chatbot_ui.launch()
            return True
            
        except Exception as e:
            logger.error(f"❌ Launch failed: {e}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            return False

def main():
    """Main entry point."""
    print("""
⚖️ Legal Document AI Assistant
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Starting setup and launch process...
    """)
    
    setup = QuickSetup()
    
    try:
        # Run setup steps
        if not setup.check_python_version():
            return False
        
        if not setup.check_structure():
            return False
        
        if not setup.install_packages():
            logger.warning("Package installation had issues but continuing...")
        
        api_ready = setup.setup_env()
        if not api_ready:
            print("\n🔑 Please configure your API key in .env file, then run: python main.py")
            return False
        
        setup.create_sample_data()
        
        if not setup.test_imports():
            print("\n❌ System not ready. Check error messages above.")
            return False
        
        # Launch
        print("\n🚀 Launching application...")
        print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        
        success = setup.launch_app()
        return success
        
    except KeyboardInterrupt:
        print("\n👋 Stopped by user")
        return False
    except Exception as e:
        logger.error(f"Setup failed: {e}")
        print(f"\n❌ Error: {e}")
        print("\n💡 Try:")
        print("1. Ensure you're in the project directory")
        print("2. Check that src/ folder exists with required files")
        print("3. Install requirements: pip install -r requirements.txt")
        return False

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)