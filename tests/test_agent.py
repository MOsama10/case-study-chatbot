"""
Tests for agent functionality.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from dataclasses import asdict

from src.agent import CaseStudyAgent, LLMWrapper, AgentResponse, answer_query


class TestLLMWrapper:
    """Test cases for LLM wrapper."""
    
    @patch('src.agent.GEMINI_API_KEY', 'test_key')
    @patch('src.agent.genai')
    def test_gemini_initialization(self, mock_genai):
        """Test Gemini LLM initialization."""
        mock_model = Mock()
        mock_genai.GenerativeModel.return_value = mock_model
        
        wrapper = LLMWrapper("gemini")
        
        assert wrapper.provider == "gemini"
        assert wrapper.client == mock_model
        mock_genai.configure.assert_called_once_with(api_key='test_key')
    
    @patch('src.agent.OPENAI_API_KEY', 'test_key')
    @patch('src.agent.OpenAI')
    def test_openai_initialization(self, mock_openai_class):
        """Test OpenAI LLM initialization."""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client
        
        wrapper = LLMWrapper("openai")
        
        assert wrapper.provider == "openai"
        assert wrapper.client == mock_client
        mock_openai_class.assert_called_once_with(api_key='test_key')
    
    def test_unsupported_provider(self):
        """Test error handling for unsupported provider."""
        with pytest.raises(ValueError, match="Unsupported LLM provider"):
            LLMWrapper("unsupported_provider")
    
    @patch('src.agent.GEMINI_API_KEY', 'test_key')
    @patch('src.agent.genai')
    def test_gemini_generate(self, mock_genai):
        """Test text generation with Gemini."""
        mock_response = Mock()
        mock_response.text = "Generated response text"
        
        mock_model = Mock()
        mock_model.generate_content.return_value = mock_response
        mock_genai.GenerativeModel.return_value = mock_model
        
        wrapper = LLMWrapper("gemini")
        result = wrapper.generate("test prompt")
        
        assert result == "Generated response text"
        mock_model.generate_content.assert_called_once()
    
    @patch('src.agent.OPENAI_API_KEY', 'test_key')
    @patch('src.agent.OpenAI')
    def test_openai_generate(self, mock_openai_class):
        """Test text generation with OpenAI."""
        mock_choice = Mock()
        mock_choice.message.content = "Generated response text"
        
        mock_response = Mock()
        mock_response.choices = [mock_choice]
        
        mock_client = Mock()
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai_class.return_value = mock_client
        
        wrapper = LLMWrapper("openai")
        result = wrapper.generate("test prompt")
        
        assert result == "Generated response text"
        mock_client.chat.completions.create.assert_called_once()


class TestCaseStudyAgent:
    """Test cases for CaseStudyAgent."""
    
    def setup_method(self):
        """Setup test environment."""
        self.mock_llm = Mock()
        self.mock_retriever = Mock()
        
        with patch('src.agent.LLMWrapper') as mock_llm_wrapper:
            mock_llm_wrapper.return_value = self.mock_llm
            
            with patch('src.agent.get_retriever') as mock_get_retriever:
                mock_get_retriever.return_value = self.mock_retriever
                
                self.agent = CaseStudyAgent()
    
    def test_build_prompt_qa_type(self):
        """Test prompt building for Q&A queries."""
        context_data = {
            'query_type': 'qa',
            'context_text': 'Test context about customer service',
            'sources': [
                {'id': 'src1', 'text': 'Source 1 content', 'score': 0.9}
            ]
        }
        
        prompt = self.agent.build_prompt("What is customer service?", context_data)
        
        assert "Test context about customer service" in prompt
        assert "What is customer service?" in prompt
        assert "Source 1 (ID: src1)" in prompt
    
    def test_build_prompt_analysis_type(self):
        """Test prompt building for analysis queries."""
        context_data = {
            'query_type': 'analysis',
            'context_text': 'Test context for analysis',
            'sources': [
                {'id': 'src1', 'text': 'Analysis source content', 'score': 0.8}
            ]
        }
        
        prompt = self.agent.build_prompt("Analyze customer trends", context_data)
        
        assert "Test context for analysis" in prompt
        assert "Analyze customer trends" in prompt
        assert "Analysis:" in prompt  # Should use analysis template
    
    def test_extract_confidence(self):
        """Test confidence extraction from responses."""
        # High confidence response
        high_conf_response = "I am certain that this solution works based on the evidence."
        confidence = self.agent.extract_confidence(high_conf_response)
        assert confidence >= 0.8
        
        # Low confidence response
        low_conf_response = "It's unclear whether this approach would work."
        confidence = self.agent.extract_confidence(low_conf_response)
        assert confidence <= 0.4
        
        # Response with sources
        sourced_response = "According to the case studies, this method is effective."
        confidence = self.agent.extract_confidence(sourced_response)
        assert confidence > 0.5
    
    @patch('src.agent.retrieve_for_query')
    def test_answer_query_success(self, mock_retrieve):
        """Test successful query answering."""
        # Mock retrieval response
        mock_retrieve.return_value = {
            'query_type': 'qa',
            'context_text': 'Context about customer service problems',
            'sources': [
                {'id': 'src1', 'text': 'Customer complaints increased', 'score': 0.9, 'source_type': 'vector'}
            ],
            'kg_results': ['Problem: Poor response time']
        }
        
        # Mock LLM response
        self.mock_llm.generate.return_value = "Customer service problems include poor response times and inadequate training."
        
        response = self.agent.answer_query("What are customer service problems?")
        
        assert isinstance(response, AgentResponse)
        assert "Customer service problems include" in response.answer
        assert len(response.sources) == 1
        assert response.sources[0]['id'] == 'src1'
        assert len(response.kg_nodes) == 1
        assert response.confidence > 0
    
    @patch('src.agent.retrieve_for_query')
    def test_answer_query_error_handling(self, mock_retrieve):
        """Test error handling in query answering."""
        # Mock retrieval to raise an exception
        mock_retrieve.side_effect = Exception("Retrieval failed")
        
        response = self.agent.answer_query("Test query")
        
        assert isinstance(response, AgentResponse)
        assert "error" in response.answer.lower()
        assert response.query_type == "error"
        assert response.confidence == 0.0


class TestAgentFunctions:
    """Test module-level agent functions."""
    
    @patch('src.agent.get_agent')
    def test_answer_query_function(self, mock_get_agent):
        """Test answer_query convenience function."""
        mock_agent = Mock()
        mock_response = AgentResponse(
            answer="Test answer",
            sources=[],
            kg_nodes=[],
            raw_llm_response="Test response",
            query_type="qa",
            confidence=0.8,
            processing_time=1.5
        )
        mock_agent.answer_query.return_value = mock_response
        mock_get_agent.return_value = mock_agent
        
        result = answer_query("test query")
        
        assert isinstance(result, dict)
        assert result['answer'] == "Test answer"
        assert result['confidence'] == 0.8
        assert result['processing_time'] == 1.5
        mock_agent.answer_query.assert_called_once_with("test query", None)
    
    @patch('src.agent.CaseStudyAgent')
    def test_get_agent_singleton(self, mock_agent_class):
        """Test that get_agent returns singleton instance."""
        mock_agent_instance = Mock()
        mock_agent_class.return_value = mock_agent_instance
        
        from src.agent import get_agent
        
        # First call
        agent1 = get_agent()
        # Second call
        agent2 = get_agent()
        
        assert agent1 is agent2  # Should be same instance
        mock_agent_class.assert_called_once()  # Should only instantiate once


class TestAgentIntegration:
    """Integration tests for agent functionality."""
    
    @patch('src.agent.retrieve_for_query')
    @patch('src.agent.LLMWrapper')
    @patch('src.agent.get_retriever')
    def test_full_pipeline_mock(self, mock_get_retriever, mock_llm_wrapper, mock_retrieve):
        """Test complete agent pipeline with mocks."""
        # Setup mocks
        mock_llm = Mock()
        mock_llm.generate.return_value = "Based on the case studies, the main issues are response time and training gaps. Source: src1"
        mock_llm_wrapper.return_value = mock_llm
        
        mock_retriever = Mock()
        mock_get_retriever.return_value = mock_retriever
        
        mock_retrieve.return_value = {
            'query_type': 'qa',
            'context_text': 'Customer service case study data',
            'sources': [
                {'id': 'src1', 'text': 'Response time is a major issue', 'score': 0.9, 'source_type': 'vector'}
            ],
            'kg_results': ['Problem: Slow response times']
        }
        
        # Test the pipeline
        response = answer_query("What are the main customer service issues?")
        
        assert isinstance(response, dict)
        assert 'main issues are response time' in response['answer']
        assert len(response['sources']) == 1
        assert response['sources'][0]['id'] == 'src1'
        assert response['confidence'] > 0


if __name__ == "__main__":
    pytest.main([__file__])