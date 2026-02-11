import os
import random
import gradio as gr
from crewai import Agent, Task, Crew, LLM

# --- 1. LLM CONFIGURATION ---
# Replace with your actual credentials or environment variables
openai_api_key = os.environ.get('OPENAI_API_KEY')

# Change to GPT-4o
gpt4o_llm = LLM(
    model="gpt-4o-mini",
    api_key=openai_api_key,
    temperature=0.4  # Lowered temperature for better factual accuracy
)

# --- 2. MULTI-AGENT CREW ---
grandma = Agent(
    role='Family Matriarch',
    goal='Generate a multiple-choice question about Lunar New Year.',
    backstory="You are the keeper of family traditions. You ask questions about food, decorations, and taboos.",
    llm=gpt4o_llm  # Changed from watsonx_llm
)

# New Agent to prevent the "Dumpling Disaster"
fact_checker = Agent(
    role='Cultural Historian',
    goal='Ensure the question is factually accurate and the correct answer is actually correct. Do NOT explain your reasoning, just provide the corrected question and answer in the same format.',
    backstory="You are an expert in East Asian studies. You verify that traditional customs are represented accurately.",
    llm=gpt4o_llm  # Changed from watsonx_llm
)

family_chat_agent = Agent(
    role='Family Group Chat Simulator',
    goal='Generate funny, short reactions from family members.',
    backstory="Xiao Ming (gamer), Auntie May (lucky), and Uncle Chen (confused) reacting to a red pocket game.",
    llm=gpt4o_llm  # Changed from watsonx_llm
)

# --- 3. CORE LOGIC ---

def generate_lunar_challenge(region):
    """Uses a 2-agent crew to verify the question before showing it."""
    task_q = Task(
        description=f"""Create a 4-option MCQ about {region} Lunar New Year. 
                        Make questions that are very easy and have no vague answers.
                        Format: Question, Options A-D, then '|||' then the letter.""",
        expected_output="Question block ||| Letter",
        agent=grandma
    )
    task_verify = Task(
        description="""
        Review the question and answer provided. 
        If it is factually incorrect (e.g., saying dumplings are NOT a Lunar New Year tradition in China), fix it. 
        If two or more options are correct, remove one or two options and replace them with clearly wrong ones.
        Ensure the hidden answer letter matches the correct option.
        Make sure the output after ||| is a SINGLE letter (A, B, C, or D) that corresponds to the correct answer.""",
        expected_output="The final verified Question block ||| Correct Letter",
        agent=fact_checker
    )
    
    crew = Crew(agents=[grandma, fact_checker], tasks=[task_q, task_verify])
    return str(crew.kickoff())

def generate_family_reactions(game_state_description):
    task_chat = Task(
        description=f"Generate 3 very short family chat messages: {game_state_description}.",
        expected_output="Three lines: 'Name: Message'",
        agent=family_chat_agent
    )
    crew = Crew(agents=[family_chat_agent], tasks=[task_chat])
    return str(crew.kickoff())

def scramble_money(total, winners):
    if not winners: return {}
    shares = []
    remaining = total
    for i in range(len(winners) - 1):
        share = round(random.uniform(0.5, (remaining / (len(winners) - i)) * 1.3), 2)
        shares.append(share)
        remaining -= share
    shares.append(round(remaining, 2))
    random.shuffle(shares)
    return dict(zip(winners, shares))

# --- 4. UI FORMATTING ---

def format_result_card(user_correct, user_gain, family_results, secret_key, total_wealth):
    status_msg = "✅ Correct!" if user_correct else "❌ Wrong!"
    status_color = "#2e7d32" if user_correct else "#b71c1c"
    
    rows = "".join([
        f"<li style='margin-bottom:2px;'>"
        f"{'✅' if d['correct'] else '❌'} {n}: {d['guess']} "
        f"{gain_html}"
        f"</li>"
        for n, d in family_results.items()
        for gain_html in [
            f"- <span style='color:#d32f2f'>+${d['gain']}</span>"
            if d['gain'] > 0 else ""
        ]
    ])
    
    return f"""
<div style="border: 2px dashed #b71c1c; border-radius: 12px; padding: 15px; background-color: #fff5f5; color: #b71c1c;">
    <h3 style="margin: 0; text-align: center;">📬 Grandma's Verdict</h3>
    <div style="text-align: center; font-weight: bold; color: {status_color};">{status_msg} (The answer was {secret_key})</div>
    <hr style="border: 0.5px solid #ffcdd2; margin: 10px 0;">
    <p style="text-align: center; margin: 5px 0;">You caught: <b>${user_gain}</b></p>
    <ul style="list-style-type: none; padding: 0; font-size: 0.85em; column-count: 2;">{rows}</ul>
    <div style="text-align: right; font-weight: bold; border-top: 1px solid #ffcdd2; padding-top: 5px; font-size: 0.9em;">Total Balance: ${total_wealth:.2f}</div>
</div>
"""

# --- 5. GAME ENGINE ---

def game_logic(user_input, region, history, state):
    if state is None: 
        state = {'status': 'IDLE', 'balance': 0, 'secret_key': '', 'chat_history': []}
    if history is None: history = []

    if state['status'] == 'IDLE':
        chat_text = generate_family_reactions(f"The family is waiting for a new question about {region}.")
        state['chat_history'].append(chat_text)
        history.append({"role": "assistant", "content": "🧧 *Grandma is consulting the ancestors (and a Fact Checker)...*"})
        yield history, state, gr.update(visible=False), gr.update(visible=False), "\n\n".join(state['chat_history'][-2:])
        
        raw_output = generate_lunar_challenge(region)
        parts = raw_output.rsplit("|||", 1)
        state['secret_key'] = parts[-1].strip() if len(parts) > 1 else "A"
        state['status'] = 'WAITING'
        
        display_q = parts[0].strip().replace('\n', '<br>')
        history[-1] = {"role": "assistant", "content": f"<div style='border: 2px solid #d4af37; border-radius: 10px; padding: 15px; background-color: #fffdf5; color: #8b0000;'>{display_q}</div>"}
        yield history, state, gr.update(visible=True, value=""), gr.update(visible=False), "\n\n".join(state['chat_history'][-2:])

    elif state['status'] == 'WAITING':
        history.append({"role": "user", "content": user_input})
        history.append({"role": "assistant", "content": "Checking red pockets... 🧧"})
        yield history, state, gr.update(visible=False), gr.update(visible=False), "💬 *Quick! Grab it!*"
        
        user_correct = user_input.strip().upper() == state['secret_key']
        winners = ["You"] if user_correct else []
        family_members_list = ["Xiao Ming", "Auntie May", "Uncle Chen"]
        
        family_data = {}
        for member in family_members_list:
            guess = random.choice(["A", "B", "C", "D"])
            is_correct = guess == state['secret_key']
            if is_correct: winners.append(member)
            family_data[member] = {"guess": guess, "correct": is_correct, "gain": 0}
            
        pot = round(random.uniform(8.88, 38.88), 2)
        payouts = scramble_money(pot, winners)
        user_gain = payouts.get("You", 0)
        state['balance'] += user_gain
        state['status'] = 'IDLE'
        
        chat_text = generate_family_reactions(f"Correct answer: {state['secret_key']}. {len(winners)} people won.")
        state['chat_history'].append(chat_text)
        for name in family_data: family_data[name]["gain"] = payouts.get(name, 0)
            
        history[-1] = {"role": "assistant", "content": format_result_card(user_correct, user_gain, family_data, state['secret_key'], state['balance'])}
        yield history, state, gr.update(visible=False), gr.update(visible=True), "\n\n".join(state['chat_history'][-2:])

# --- 6. UI ASSEMBLY ---

custom_css = """
.gradio-container { background-color: #fdf5e6; font-family: 'Georgia', serif; }
#chat-sidebar { background: #fffdf5; border: 1px solid #d4af37; border-radius: 8px; padding: 10px; min-height: 350px; color: #8b0000; font-size: 0.85em; white-space: pre-wrap; }
#balance-box { background: #b71c1c; color: #ffd700; padding: 10px; border-radius: 8px; text-align: center; font-weight: bold; border: 2px solid #d4af37; font-size: 18px; }
"""

with gr.Blocks(css=custom_css) as demo:
    gr.HTML("<h1 style='text-align: center; color: #8b0000; margin-bottom:0;'>🏮 The Fact-Checked Scramble 🏮</h1>")
    state = gr.State()
    
    with gr.Row():
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(height=450)
            with gr.Row():
                ans_msg = gr.Textbox(placeholder="Letter (A, B, C, or D)...", visible=False, scale=4)
                start_btn = gr.Button("🧧 New Round", variant="primary")
                submit_btn = gr.Button("Submit Answer", variant="stop", visible=False)
        
        with gr.Column(scale=1):
            region_drop = gr.Dropdown(choices=["Mainland China", "Vietnam", "Korea", "North America"], label="Region", value="Mainland China")
            balance_display = gr.HTML("<div id='balance-box'>Luck: $0.00</div>")
            gr.Markdown("#### 👨‍👩‍👧‍👦 Family Chat")
            family_chat_box = gr.Markdown(value="*Group chat joined...*", elem_id="chat-sidebar")

    start_btn.click(game_logic, [gr.State(""), region_drop, chatbot, state], [chatbot, state, ans_msg, start_btn, family_chat_box]).then(lambda s: f"<div id='balance-box'>Luck: ${s['balance']:.2f}</div>", state, balance_display)
    submit_btn.click(game_logic, [ans_msg, region_drop, chatbot, state], [chatbot, state, ans_msg, start_btn, family_chat_box]).then(lambda s: f"<div id='balance-box'>Luck: ${s['balance']:.2f}</div>", state, balance_display)
    start_btn.click(lambda: (gr.update(visible=True), gr.update(visible=False)), None, [submit_btn, start_btn])
    submit_btn.click(lambda: (gr.update(visible=False), gr.update(visible=True)), None, [submit_btn, start_btn])

if __name__ == "__main__":
    demo.launch()
