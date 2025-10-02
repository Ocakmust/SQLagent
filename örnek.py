# Reasoning Patterns - Prompt Örnekleri

## 1. Chain-of-Thought (CoT)

### Temel CoT Prompt
```
Question: Roger has 5 tennis balls. He buys 2 more cans of tennis balls. 
Each can has 3 tennis balls. How many tennis balls does he have now?

Let's think step by step:
```

**Model Response:**
```
1. Roger starts with 5 tennis balls
2. He buys 2 cans of tennis balls
3. Each can contains 3 tennis balls
4. So from the cans: 2 × 3 = 6 tennis balls
5. Total: 5 + 6 = 11 tennis balls

Answer: Roger has 11 tennis balls.
```

---

### CoT ile Zero-Shot (Talimatla)
```
Solve this problem step by step, showing your reasoning:

Question: A restaurant had 23 customers. 14 more customers arrived. 
Then 8 customers left. How many customers are there now?
```

---

### CoT ile Few-Shot (Örnekle Öğretme)
```
Question: Olivia has $23. She bought five bagels for $3 each. How much money does she have left?
Answer: Let's think step by step.
- Olivia had $23
- 5 bagels cost: 5 × $3 = $15
- Money left: $23 - $15 = $8
Answer: $8

Question: John has 15 apples. He gives 4 to Mary and 3 to Bob. How many does he have left?
Answer: Let's think step by step.
- John started with 15 apples
- He gave away: 4 + 3 = 7 apples
- Remaining: 15 - 7 = 8 apples
Answer: 8

Question: Sarah has $50. She buys 3 books for $12 each. How much money does she have left?
Answer: Let's think step by step.
```

---

## 2. Self-Consistency

### Self-Consistency Prompt (Aynı Soruyu Birden Fazla Kez)

**Sampling 1:**
```
Question: If there are 3 cars in the parking lot and 2 more cars arrive, 
how many cars are in the parking lot?

Let's think step by step:

Starting cars: 3
Cars that arrived: 2
Total: 3 + 2 = 5 cars

Answer: 5
```

**Sampling 2:**
```
Question: If there are 3 cars in the parking lot and 2 more cars arrive, 
how many cars are in the parking lot?

Let's solve this carefully:

We begin with 3 cars already parked.
2 additional cars come.
Adding them: 3 + 2 = 5

Answer: 5
```

**Sampling 3:**
```
Question: If there are 3 cars in the parking lot and 2 more cars arrive, 
how many cars are in the parking lot?

Let me work through this:

Initial count: 3 vehicles
New arrivals: 2 vehicles
Sum: 3 plus 2 equals 5

Answer: 5
```

**Majority Vote → Final Answer: 5 ✓**

---

### Self-Consistency Implementation Pseudo-Code
```python
# Self-Consistency Pattern
def self_consistency_solve(question, n_samples=5, temperature=0.7):
    answers = []
    
    for i in range(n_samples):
        # Her seferinde temperature > 0 ile farklı reasoning path
        prompt = f"{question}\n\nLet's think step by step:"
        response = llm.generate(prompt, temperature=temperature)
        
        # Response'dan final answer'ı extract et
        answer = extract_answer(response)
        answers.append(answer)
    
    # Majority voting
    final_answer = most_common(answers)
    return final_answer

# Örnek kullanım
question = "Roger has 5 balls. He buys 2 cans with 3 balls each. How many total?"
answer = self_consistency_solve(question, n_samples=5)
# → 5 farklı reasoning path dener, en çok tekrar edeni seçer
```

---

## 3. Tree of Thoughts (ToT)

### ToT Prompt - Game of 24 Örneği
```
Use numbers 4, 9, 10, 13 and basic operations (+, -, *, /) to get 24.
Each number must be used exactly once.

Let me explore multiple solution paths:

Path 1: Try multiplication first
- 13 - 9 = 4
- 4 * 4 = 16
- 16 + 10 = 26 ❌ (too high)

Path 2: Try different grouping
- (13 - 9) * (10 - 4) = ?
- 4 * 6 = 24 ✓

Path 3: Try addition/subtraction
- 13 + 9 + 4 - 10 = ?
- 22 + 4 - 10 = 16 ❌

Best solution found: (13 - 9) * (10 - 4) = 24
```

---

### ToT Prompt - Creative Writing Örneği
```
Write a short story about a robot learning to feel emotions.
Generate 3 different opening paragraphs, evaluate each, then continue with the best:

Opening 1:
"Unit-7 had calculated probabilities for 247 years, but today's equation yielded 
an impossible result: loneliness."
[Evaluation: Strong, intriguing, sets up conflict]

Opening 2:
"The laboratory was cold and metallic, just like the robot standing in its center."
[Evaluation: Generic, lacks hook]

Opening 3:
"She didn't expect the malfunction. None of them did. But when Unit-7's emotion 
subroutine activated, everything changed."
[Evaluation: Good tension, but unclear]

Selected: Opening 1 (highest score)
Now continue the story...
```

---

### ToT Implementation Pattern
```python
# Tree of Thoughts Pattern
def tree_of_thoughts(problem, depth=3, breadth=3):
    """
    depth: Kaç seviye derine in
    breadth: Her seviyede kaç alternatif dene
    """
    
    def generate_thoughts(state, level):
        """Her state için 'breadth' kadar thought üret"""
        thoughts = []
        for i in range(breadth):
            prompt = f"Current state: {state}\nGenerate next step {i+1}:"
            thought = llm.generate(prompt)
            thoughts.append(thought)
        return thoughts
    
    def evaluate_thought(thought):
        """Thought'ı 1-10 arası score et"""
        prompt = f"Evaluate this reasoning step (1-10):\n{thought}"
        score = llm.generate(prompt)
        return float(score)
    
    # BFS: Breadth-first search
    current_states = [{"state": problem, "path": []}]
    
    for level in range(depth):
        next_states = []
        
        for state in current_states:
            # Her state için 'breadth' kadar thought üret
            thoughts = generate_thoughts(state["state"], level)
            
            for thought in thoughts:
                # Her thought'ı evaluate et
                score = evaluate_thought(thought)
                
                # Yeni state oluştur
                new_state = {
                    "state": thought,
                    "path": state["path"] + [thought],
                    "score": score
                }
                next_states.append(new_state)
        
        # En iyi 'breadth' kadar state'i tut (pruning)
        current_states = sorted(next_states, key=lambda x: x["score"], reverse=True)[:breadth]
    
    # En iyi path'i döndür
    best_solution = current_states[0]
    return best_solution["path"]
```

---

## 4. ReAct (Reasoning + Acting)

### ReAct Prompt Template
```
You are an assistant that can use tools to help answer questions.

Available tools:
- search(query: str) -> str: Search the web
- calculator(expression: str) -> float: Calculate math expressions
- get_weather(city: str) -> str: Get current weather

Answer the following question using this format:

Thought: [Your reasoning about what to do next]
Action: [The tool to use: search, calculator, or get_weather]
Action Input: [The input to the tool]
Observation: [The result from the tool will appear here]
... (repeat Thought/Action/Observation as needed)
Thought: [Final reasoning]
Final Answer: [Your final answer to the question]

Question: What is the weather like in San Francisco and what's 25% of the temperature in Fahrenheit?
```

---

### ReAct Execution Example
```
Question: What is the weather like in San Francisco and what's 25% of the temperature in Fahrenheit?

Thought: I need to first find out the current weather in San Francisco, 
particularly the temperature.

Action: get_weather
Action Input: San Francisco

Observation: The current weather in San Francisco is partly cloudy with a 
temperature of 68°F.

Thought: Now I have the temperature (68°F). I need to calculate 25% of 68.

Action: calculator
Action Input: 68 * 0.25

Observation: 17.0

Thought: I now have all the information needed to answer the question. 
The weather is partly cloudy at 68°F, and 25% of that temperature is 17°F.

Final Answer: The weather in San Francisco is partly cloudy with a temperature 
of 68°F. 25% of this temperature is 17°F.
```

---

### ReAct - Multi-Step Research Example
```
Question: Who was the president of the United States when the first iPhone was released, 
and what was their approval rating at that time?

Thought: I need to find out when the first iPhone was released.

Action: search
Action Input: when was the first iPhone released

Observation: The first iPhone was released on June 29, 2007.

Thought: Now I know it was released in 2007. I need to find who was the U.S. 
president in June 2007.

Action: search
Action Input: US president June 2007

Observation: George W. Bush was the President of the United States in June 2007.

Thought: Now I need to find George W. Bush's approval rating in June 2007.

Action: search
Action Input: George W Bush approval rating June 2007

Observation: According to polls, George W. Bush's approval rating in June 2007 
was approximately 29-32%.

Thought: I have all the information needed to provide a complete answer.

Final Answer: George W. Bush was the President of the United States when the first 
iPhone was released on June 29, 2007. His approval rating at that time was 
approximately 29-32%.
```

---

### ReAct Implementation (Python-like Pseudocode)
```python
def react_agent(question, tools, max_iterations=10):
    """
    ReAct pattern implementation
    """
    context = f"Question: {question}\n\n"
    
    for i in range(max_iterations):
        # 1. LLM'den Thought + Action iste
        prompt = f"""
{context}

What should you do next? Use this format:
Thought: [reasoning]
Action: [tool_name]
Action Input: [input]
"""
        
        response = llm.generate(prompt)
        
        # 2. Response'u parse et
        thought = extract_thought(response)
        action = extract_action(response)
        action_input = extract_action_input(response)
        
        context += f"Thought: {thought}\n"
        context += f"Action: {action}\n"
        context += f"Action Input: {action_input}\n"
        
        # 3. Check if finished
        if action == "Final Answer":
            return action_input
        
        # 4. Execute the tool
        tool_result = execute_tool(action, action_input, tools)
        
        context += f"Observation: {tool_result}\n\n"
    
    return "Max iterations reached without final answer"

# Örnek kullanım
tools = {
    "search": web_search_function,
    "calculator": calculator_function,
    "get_weather": weather_api_function
}

answer = react_agent(
    "What's the weather in Paris and how does it compare to London?",
    tools
)
```

