#!/usr/bin/env python3
"""
Comprehensive setup and fix script for the Case Study Chatbot.
This script will:
1. Create missing files
2. Fix import issues
3. Set up sample data
4. Validate the system
5. Launch the enhanced UI
"""

import os
import sys
import subprocess
import logging
from pathlib import Path
from typing import List, Dict, Any
import json

def setup_logging():
    """Configure logging for the setup process."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('setup.log')
        ]
    )
    return logging.getLogger(__name__)

logger = setup_logging()

class ComprehensiveSetup:
    """Comprehensive setup manager for the chatbot project."""
    
    def __init__(self):
        """Initialize the setup manager."""
        self.project_root = Path.cwd()
        self.src_dir = self.project_root / "src"
        self.data_dir = self.project_root / "data"
        self.issues_found = []
        self.fixes_applied = []
    
    def check_python_version(self) -> bool:
        """Check if Python version is compatible."""
        logger.info("🐍 Checking Python version...")
        
        version = sys.version_info
        if version.major < 3 or (version.major == 3 and version.minor < 10):
            logger.error(f"❌ Python 3.10+ required, found {version.major}.{version.minor}")
            return False
        
        logger.info(f"✅ Python {version.major}.{version.minor}.{version.micro} is compatible")
        return True
    
    def install_dependencies(self) -> bool:
        """Install required dependencies."""
        logger.info("📦 Installing dependencies...")
        
        try:
            # Check if requirements.txt exists
            req_file = self.project_root / "requirements.txt"
            if not req_file.exists():
                logger.warning("⚠️ requirements.txt not found, creating it...")
                self.create_requirements_file()
            
            # Install dependencies
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", "-r", str(req_file)
            ])
            
            logger.info("✅ Dependencies installed successfully")
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Failed to install dependencies: {e}")
            return False
    
    def create_requirements_file(self):
        """Create requirements.txt with all necessary dependencies."""
        requirements = """# Core dependencies
python-dotenv>=1.0.0
pathlib2>=2.3.7
typing-extensions>=4.5.0

# Document processing
python-docx>=0.8.11
beautifulsoup4>=4.12.0

# ML/AI - Core libraries
sentence-transformers>=2.2.2
faiss-cpu>=1.7.4
numpy>=1.24.0
scipy>=1.10.0
scikit-learn>=1.3.0

# Knowledge graph
networkx>=3.1.0

# LLM APIs
google-generativeai>=0.3.0
openai>=1.3.0

# Web UI
gradio>=3.50.0

# Data processing
pandas>=2.0.0
requests>=2.31.0
tqdm>=4.65.0

# Utilities and formatting
rich>=13.5.0
colorama>=0.4.6

# Testing
pytest>=7.4.0
pytest-mock>=3.11.0
"""
        
        req_file = self.project_root / "requirements.txt"
        req_file.write_text(requirements)
        logger.info("✅ Created requirements.txt")
    
    def create_missing_files(self) -> bool:
        """Create any missing essential files."""
        logger.info("📄 Creating missing files...")
        
        # Create .env file if it doesn't exist
        env_file = self.project_root / ".env"
        if not env_file.exists():
            env_content = """# API Configuration - Choose one:

# Gemini API (Free tier available)
GEMINI_API_KEY=your_gemini_api_key_here

# OpenAI API (Paid)
# OPENAI_API_KEY=your_openai_api_key_here
# LLM_PROVIDER=openai

# Default configuration
LLM_PROVIDER=gemini
GEMINI_MODEL=models/gemini-pro-latest

# Performance settings
EMBEDDING_BATCH_SIZE=16
VECTOR_TOP_K=8
"""
            env_file.write_text(env_content)
            logger.info("✅ Created .env file template")
            self.fixes_applied.append("Created .env template")
        
        # Create __init__.py files
        init_files = [
            self.src_dir / "__init__.py",
            self.project_root / "tests" / "__init__.py"
        ]
        
        for init_file in init_files:
            if not init_file.exists():
                init_file.parent.mkdir(parents=True, exist_ok=True)
                init_file.write_text('"""Package initialization."""\n')
                logger.info(f"✅ Created {init_file}")
                self.fixes_applied.append(f"Created {init_file}")
        
        return True
    
    def create_sample_data(self) -> bool:
        """Create sample data for testing."""
        logger.info("📚 Creating sample data...")
        
        try:
            # Create data directories
            self.data_dir.mkdir(exist_ok=True)
            (self.data_dir / "batch_1").mkdir(exist_ok=True)
            
            # Check if we have the sample data creator
            sample_creator = self.project_root / "create_sample_data.py"
            if sample_creator.exists():
                # Run the sample data creator
                subprocess.run([sys.executable, str(sample_creator)], check=True)
                logger.info("✅ Sample data created using create_sample_data.py")
            else:
                # Create basic sample data manually
                self.create_basic_sample_data()
                logger.info("✅ Basic sample data created manually")
            
            self.fixes_applied.append("Created sample data")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to create sample data: {e}")
            return False
    
    def create_basic_sample_data(self):
        """Create basic sample data manually if docx creation fails."""
        # Create a simple text-based data structure that the system can use
        sample_data = {
            "case_studies": [
                {
                    "id": "cs_001",
                    "title": "Customer Service Improvement",
                    "problem": "High response times and low customer satisfaction scores",
                    "solution": "Implemented automated ticketing and additional training",
                    "result": "92% improvement in response time, customer satisfaction increased to 89%"
                },
                {
                    "id": "cs_002", 
                    "title": "Employee Retention Initiative",
                    "problem": "45% annual turnover rate causing disruption and costs",
                    "solution": "Career development programs and competitive compensation review",
                    "result": "Turnover reduced to 15%, saving $2.1M in recruitment costs"
                },
                {
                    "id": "cs_003",
                    "title": "Digital Transformation",
                    "problem": "Legacy systems limiting growth and customer experience",
                    "solution": "Omnichannel platform with mobile app and integrated inventory",
                    "result": "Online sales grew to 35% of revenue, overall sales up 22%"
                }
            ]
        }
        
        # Save as JSON for fallback processing
        fallback_file = self.data_dir / "fallback_cases.json"
        fallback_file.write_text(json.dumps(sample_data, indent=2))
        logger.info("✅ Created fallback sample data as JSON")
    
    def validate_imports(self) -> bool:
        """Validate that all imports work correctly."""
        logger.info("🔍 Validating imports...")
        
        try:
            # Test core imports
            sys.path.insert(0, str(self.project_root))
            
            # Test configuration
            from src.config import get_logger
            logger.info("✅ Config module imports working")
            
            # Test data loader
            from src.data_loader import preprocess_text, chunk_text
            logger.info("✅ Data loader imports working")
            
            # Test embeddings
            from src.embeddings import EmbeddingManager
            logger.info("✅ Embeddings imports working")
            
            # Test knowledge graph
            from src.knowledge_graph import get_knowledge_graph
            logger.info("✅ Knowledge graph imports working")
            
            # Test retriever
            from src.retriever import get_retriever
            logger.info("✅ Retriever imports working")
            
            # Test agent
            from src.agent import get_agent
            logger.info("✅ Agent imports working")
            
            # Test chatbot
            from src.chatbot import handle_message
            logger.info("✅ Chatbot imports working")
            
            self.fixes_applied.append("All imports validated")
            return True
            
        except ImportError as e:
            logger.error(f"❌ Import error: {e}")
            self.issues_found.append(f"Import error: {e}")
            return False
        except Exception as e:
            logger.error(f"❌ Validation error: {e}")
            self.issues_found.append(f"Validation error: {e}")
            return False
    
    def check_api_configuration(self) -> bool:
        """Check if API keys are configured."""
        logger.info("🔑 Checking API configuration...")
        
        from dotenv import load_dotenv
        load_dotenv()
        
        gemini_key = os.getenv("GEMINI_API_KEY")
        openai_key = os.getenv("OPENAI_API_KEY")
        
        if not gemini_key and not openai_key:
            logger.warning("⚠️ No API keys configured")
            logger.info("📝 Please edit the .env file and add your API key")
            return False
        
        if gemini_key and gemini_key != "your_gemini_api_key_here":
            logger.info("✅ Gemini API key configured")
            return True
        
        if openai_key and openai_key != "your_openai_api_key_here":
            logger.info("✅ OpenAI API key configured")
            return True
        
        logger.warning("⚠️ API keys found but appear to be placeholders")
        return False
    
    def test_system_functionality(self) -> bool:
        """Test basic system functionality."""
        logger.info("🧪 Testing system functionality...")
        
        try:
            # Test embeddings
            from src.embeddings import get_embedding_manager
            manager = get_embedding_manager()
            logger.info("✅ Embedding manager initialized")
            
            # Test knowledge graph
            from src.knowledge_graph import get_knowledge_graph
            kg = get_knowledge_graph()
            logger.info(f"✅ Knowledge graph loaded with {kg.number_of_nodes()} nodes")
            
            # Test agent with a simple query
            from src.agent import get_agent
            agent = get_agent()
            logger.info("✅ Agent initialized")
            
            # Simple test query
            from src.chatbot import handle_message
            response = handle_message("test_user", "Hello")
            if response and response.get('answer'):
                logger.info("✅ End-to-end test successful")
                return True
            else:
                logger.warning("⚠️ End-to-end test returned empty response")
                return False
                
        except Exception as e:
            logger.error(f"❌ System test failed: {e}")
            self.issues_found.append(f"System test failed: {e}")
            return False
    
    def generate_report(self) -> str:
        """Generate a setup report."""
        report = """
🚀 SETUP COMPLETE - CASE STUDY CHATBOT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

"""
        if self.fixes_applied:
            report += "✅ FIXES APPLIED:\n"
            for fix in self.fixes_applied:
                report += f"   • {fix}\n"
            report += "\n"
        
        if self.issues_found:
            report += "⚠️ ISSUES FOUND:\n"
            for issue in self.issues_found:
                report += f"   • {issue}\n"
            report += "\n"
        
        report += """🎯 NEXT STEPS:
1. Edit .env file and add your API key (Gemini or OpenAI)
2. Run the application: python src/ui.py
3. Open browser to http://localhost:7860
4. Start asking questions about business case studies!

💡 EXAMPLE QUERIES:
• "What are effective customer retention strategies?"
• "Analyze digital transformation success factors"  
• "How do companies reduce operational costs?"

🆘 NEED HELP?
• Check the .env file for API key configuration
• View setup.log for detailed information
• Ensure you have Python 3.10+ installed

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
        return report
    
    def run_complete_setup(self) -> bool:
        """Run the complete setup process."""
        logger.info("🚀 Starting comprehensive setup...")
        
        success = True
        
        # Step 1: Check Python version
        if not self.check_python_version():
            return False
        
        # Step 2: Install dependencies
        if not self.install_dependencies():
            success = False
        
        # Step 3: Create missing files
        if not self.create_missing_files():
            success = False
        
        # Step 4: Create sample data
        if not self.create_sample_data():
            success = False
        
        # Step 5: Validate imports
        if not self.validate_imports():
            success = False
        
        # Step 6: Check API configuration
        api_configured = self.check_api_configuration()
        if not api_configured:
            self.issues_found.append("API keys not configured - edit .env file")
        
        # Step 7: Test system (only if API is configured)
        if api_configured:
            if not self.test_system_functionality():
                success = False
        
        # Generate and display report
        report = self.generate_report()
        print(report)
        
        return success


def main():
    """Main setup function."""
    print("""
🚀 Case Study Chatbot - Comprehensive Setup
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

This script will fix all issues and get your chatbot running!
    """)
    
    setup = ComprehensiveSetup()
    
    try:
        success = setup.run_complete_setup()
        
        if success:
            print("\n🎉 Setup completed successfully!")
            
            # Ask if user wants to launch the UI
            launch = input("\n🚀 Launch the chatbot now? (y/n): ").lower().strip()
            if launch in ['y', 'yes']:
                print("\n🔥 Launching enhanced UI...")
                try:
                    from src.ui import main as ui_main
                    ui_main()
                except KeyboardInterrupt:
                    print("\n👋 Goodbye!")
                except Exception as e:
                    print(f"\n❌ UI launch failed: {e}")
                    print("💡 Try running manually: python src/ui.py")
        else:
            print("\n⚠️ Setup completed with some issues.")
            print("📋 Check the report above and setup.log for details.")
            
    except KeyboardInterrupt:
        print("\n🛑 Setup interrupted by user")
    except Exception as e:
        logger.error(f"❌ Setup failed with error: {e}")
        print(f"\n💥 Setup failed: {e}")
        print("📋 Check setup.log for detailed error information")


if __name__ == "__main__":
    main()