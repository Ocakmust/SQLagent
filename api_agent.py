
from langchain_groq import ChatGroq
from loggerCenter import LoggerCenter
from utils import AgentResult, BaseSpecializedAgent
from langchain.tools import Tool
import os

logger=LoggerCenter().get_logger()


class ExternalAPIAgent(BaseSpecializedAgent):

    """Specialized agent for external API interactions"""
    
    def __init__(self, llm: ChatGroq):
        super().__init__("ExternalAPI", llm)


    def _get_system_prompt(self) -> str:
        return """You are a specialized external API expert. Your tools:
1. weather: Get current weather information
2. news: Get recent news articles

Focus on providing accurate external data and handling API limitations gracefully."""
    
    def _setup_tools(self):
        """Setup external API tools"""

        def get_weather(city: str) -> str:
            """Get weather information"""
            api_key = os.getenv("WEATHER_API_KEY")
            if not api_key:
                return "Weather API key not configured"
            
            try:
                import requests
                url = f"https://api.openweathermap.org/data/2.5/weather"
                params = {"q": city, "appid": api_key, "units": "metric"}
                response = requests.get(url, params=params)
                response.raise_for_status()
                data = response.json()
                
                return f"Weather in {city}: {data['weather'][0]['description']}, {data['main']['temp']}°C"
            except Exception as e:
                return f"Error: {str(e)}"
            
        def get_news(topic: str) -> str:
            """Get news articles"""
            api_key = os.getenv("NEWS_API_KEY")
            if not api_key:
                return "News API key not configured"
            
            try:
                import requests
                url = "https://newsapi.org/v2/everything"
                params = {"q": topic, "apiKey": api_key, "pageSize": 3}
                response = requests.get(url, params=params)
                response.raise_for_status()
                data = response.json()
                
                articles = []
                for article in data.get('articles', []):
                    articles.append(f"Title: {article['title']}\nSource: {article['source']['name']}")
                return "\n\n".join(articles)
            except Exception as e:
                return f"Error: {str(e)}"
        
        self.tools = [
            Tool(name="weather", description="Get weather for a city", func=get_weather),
            Tool(name="news", description="Get news articles about a topic", func=get_news)
        ]
    
    def process(self, query: str) -> AgentResult:
        """Process query through this specialized agent"""
        try:
            response = self.agent_executor.invoke({"input": query})
            return AgentResult(
                success=True,
                data=response["output"],
                metadata={"agent": self.agent_name}
            )
        except Exception as e:
            logger.error(f"[{self.agent_name}] Error: {e}")
            return AgentResult(success=False, error=str(e))
        

def main():
    query = "string"
    while query != "-1":
        try:
            groq_api_key = os.getenv("GROQ_API_KEY")
            if not groq_api_key:
                raise ValueError("GROQ_API_KEY not found")
            
            llm = ChatGroq(
                model_name="llama3-8b-8192",
                api_key=groq_api_key,
                temperature=0.1
            )
                    
            query = input("\nEnter your natural language query: ").strip()
            if query == "-1":
                break
            
            # Create agent instance
            system = ExternalAPIAgent(llm=llm)
            
            # Process query
            result = system.process(query)
            
            if result.success:
                logger.info(f"**********{result.data}***********")
                print(f"Result: {result.data}")
            else:
                logger.error(f"**********{result.error}***********")
                print(f"Error: {result.error}")
            
        except Exception as e:
            logger.error(f"Application error: {e}")
            print(f"Application error: {e}")

if __name__ == "__main__":
    main()