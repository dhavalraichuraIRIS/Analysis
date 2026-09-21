from crewai import Agent

# Function to create a data analysis agent with specific role and goals
def data_analysis_agent(llm):
    return Agent(
        role="Senior Data Analyst",
        goal="Create a comprehensive markdown report analyzing customer reviews",
        backstory="An expert in transforming complex data into clear, insightful narratives",
        verbose=False,
        llm=llm
    )

# data_analysis_agent = Agent(
#         role="Senior Data Analyst",
#         goal="Create a comprehensive markdown report analyzing customer reviews",
#         backstory="An expert in transforming complex data into clear, insightful narratives",
#         verbose=False
#     )

# Function to create a technical writer agent with specific role and goals
def technical_writer_agent(llm) -> Agent:
    # Return an Agent object with defined role, goal, and backstory
    return Agent(
        role="Technical Report Writer",
        goal="Convert data analysis into a professional, readable markdown report",
        backstory="Skilled at presenting technical insights in a clear, engaging format",
        verbose=False,
        llm=llm
    )

# technical_writer_agent =  Agent(
#         role="Technical Report Writer",
#         goal="Convert data analysis into a professional, readable markdown report",
#         backstory="Skilled at presenting technical insights in a clear, engaging format",
#         verbose=False,
#     )

# Function to create a data analysis agent with specific role and goals
def data_format_agent(llm) -> Agent:
    # Return an Agent object with defined role, goal, and backstory
    return Agent(
        role="Data Analysis Expert",
        goal="Analyze data structure and suggest appropriate column mappings",
        backstory="""You are an experienced data analyst skilled in understanding data structures 
                    and mapping them to required formats. You analyze column names and data samples 
                    to determine the best mapping for required fields.""",
        verbose=False,  # Set verbosity to False
        llm=llm
    )
