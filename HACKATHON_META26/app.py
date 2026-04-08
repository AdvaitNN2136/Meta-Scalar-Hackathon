import gradio as gr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from stable_baselines3 import PPO
from env import DisasterEnv

model = None
custom_objects = {
    "learning_rate": 0.0,
    "lr_schedule": lambda _: 0.0,
    "clip_range": lambda _: 0.2
}

for path in ["disaster_ppo_final", "models/best/best_model", "models/disaster_ppo_400000_steps"]:
    try:
        model = PPO.load(path, custom_objects=custom_objects)
        print(f"Loaded: {path}")
        break
    except Exception as e:
        print(f"Skip {path}: {e}")

DISASTER_COLORS = {
    "none":       "#2ecc71",
    "fire":       "#e74c3c",
    "flood":      "#3498db",
    "earthquake": "#e67e22",
    "lightning":  "#f1c40f",
    "tsunami":    "#1abc9c",
    "compound":   "#8e44ad",
    "epidemic":   "#c0392b",
    "aftershock": "#d35400",
}

WEATHER_EMOJI = {
    "clear":    "☀ Clear",
    "rain":     "🌧 Rain",
    "storm":    "⛈ Storm",
    "heatwave": "🌡 Heatwave",
}

def run_episode(use_agent=True, speed=1.0):
    env = DisasterEnv()
    obs, _ = env.reset()
    history = []

    for _ in range(env.max_steps):
        if use_agent:
            action, _ = model.predict(obs, deterministic=True)
        else:
            action = env.action_space.sample()

        obs, reward, done, truncated, info = env.step(int(action))
        history.append({
            "severity":        info["severity"].copy(),
            "disaster_names":  info["disaster_names"].copy(),
            "city_names":      info["city_names"],
            "compound_flag":   info["compound_flag"].copy(),
            "power_out":       info["power_out"].copy(),
            "disease_risk":    info["disease_risk"].copy(),
            "infrastructure":  info["infrastructure_dmg"].copy(),
            "teams":           info["teams_available"],
            "saved":           info["total_saved"],
            "casualties":      info["total_casualties"],
            "supply":          info["supply_level"],
            "weather":         info["weather"],
            "time":            info["time_of_day"],
            "cascades":        info["cascade_events"],
        })
        if done or truncated:
            break

    return history

def make_figure(history):
    n_cities = len(history[0]["city_names"])
    city_names = history[0]["city_names"]
    steps = len(history)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.patch.set_facecolor("#0f0f1a")
    for ax in axes.flat:
        ax.set_facecolor("#1a1a2e")
        ax.tick_params(colors="white")
        ax.title.set_color("white")
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#333")

    # Plot 1: Severity over time per city
    ax1 = axes[0, 0]
    colors = plt.cm.tab10(np.linspace(0, 1, n_cities))
    for i, (city, color) in enumerate(zip(city_names, colors)):
        sev_over_time = [h["severity"][i] for h in history]
        ax1.plot(sev_over_time, label=city, color=color, linewidth=1.5)
    ax1.set_title("Severity Over Time Per City", fontsize=12, fontweight="bold")
    ax1.set_ylim(0, 1)
    ax1.set_xlabel("Step")
    ax1.set_ylabel("Severity")
    ax1.legend(loc="upper right", fontsize=7, facecolor="#1a1a2e", labelcolor="white")
    ax1.axhline(0.7, color="red", linestyle="--", alpha=0.4, linewidth=0.8)
    ax1.text(1, 0.72, "Cascade threshold", color="red", fontsize=7, alpha=0.7)

    # Plot 2: Final state bar chart
    ax2 = axes[0, 1]
    final = history[-1]
    bar_colors = [DISASTER_COLORS.get(d, "#888") for d in final["disaster_names"]]
    bars = ax2.bar(city_names, final["severity"], color=bar_colors, alpha=0.9, edgecolor="#333")
    ax2.set_title("Final Severity by City", fontsize=12, fontweight="bold")
    ax2.set_ylim(0, 1)
    ax2.set_ylabel("Severity")
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=30, ha="right", fontsize=8)
    for bar, name in zip(bars, final["disaster_names"]):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                 name[:3].upper(), ha="center", va="bottom",
                 fontsize=7, color="white", fontweight="bold")

    # Plot 3: Disease risk + infrastructure damage
    ax3 = axes[1, 0]
    x = np.arange(n_cities)
    w = 0.35
    ax3.bar(x - w/2, final["disease_risk"],   width=w, label="Disease risk",    color="#c0392b", alpha=0.8)
    ax3.bar(x + w/2, final["infrastructure"], width=w, label="Infra damage",    color="#e67e22", alpha=0.8)
    ax3.set_xticks(x)
    ax3.set_xticklabels(city_names, rotation=30, ha="right", fontsize=8)
    ax3.set_title("Disease Risk & Infrastructure Damage", fontsize=12, fontweight="bold")
    ax3.set_ylim(0, 1)
    ax3.legend(facecolor="#1a1a2e", labelcolor="white", fontsize=8)

    # Plot 4: Saved vs Casualties + supply over time
    ax4 = axes[1, 1]
    saved_over_time = [h["saved"] for h in history]
    cas_over_time   = [h["casualties"] for h in history]
    supply_over_time= [h["supply"] * max(saved_over_time) for h in history]
    ax4.plot(saved_over_time, color="#2ecc71", linewidth=2, label="People saved")
    ax4.plot(cas_over_time,   color="#e74c3c", linewidth=2, label="Casualties")
    ax4.plot(supply_over_time,color="#f1c40f", linewidth=1.5,
             linestyle="--", label="Supply (scaled)")
    ax4.set_title("Saved vs Casualties Over Time", fontsize=12, fontweight="bold")
    ax4.set_xlabel("Step")
    ax4.legend(facecolor="#1a1a2e", labelcolor="white", fontsize=8)

    plt.tight_layout()
    return fig

def run_and_plot(mode):
    use_agent = (mode == "RL Agent (PPO)")
    history   = run_episode(use_agent)
    fig       = make_figure(history)
    final     = history[-1]

    # Cascade log
    all_cascades = []
    for h in history:
        all_cascades.extend(h["cascades"])

    summary = f"""
SIMULATION COMPLETE
Mode         : {mode}
Duration     : {len(history)} steps ({len(history)} hours simulated)
Weather      : {WEATHER_EMOJI.get(final['weather'], final['weather'])}
Time         : {final['time']}

RESULTS
People Saved    : {final['saved']:,}
Total Casualties: {final['casualties']:,}
Supply Level    : {final['supply']:.0%}
Avg Severity    : {np.mean(final['severity']):.2f}

CASCADE EVENTS ({len(all_cascades)} total)
""" + "\n".join(all_cascades[-10:]) if all_cascades else "No cascades triggered"

    return fig, summary

with gr.Blocks(
    title="Disaster Response RL",
    css="""
    body { background: #0f0f1a; }
    .gradio-container { background: #0f0f1a; color: white; }
    """
) as demo:
    gr.Markdown("""
    # Disaster Response Management System
    ### Reinforcement Learning Agent vs Random Baseline
    **Cities**: Chennai, Mumbai, Delhi, Bengaluru, Mysuru, Hyderabad, Kochi, Coorg
    **Disasters**: Fire, Flood, Earthquake→Tsunami, Lightning→Compound, Epidemic, Aftershocks
    **Systems**: Weather, Day/Night cycle, Power grid, Supply chain, Team fatigue, Infrastructure
    """)

    with gr.Row():
        mode_dd  = gr.Dropdown(
            choices=["RL Agent (PPO)", "Random Baseline"],
            value="RL Agent (PPO)",
            label="Select Mode"
        )
        run_btn  = gr.Button("Run Simulation", variant="primary", scale=2)

    plot_out    = gr.Plot(label="Simulation Results")
    summary_out = gr.Textbox(label="Summary & Cascade Log",
                             lines=20, max_lines=30)

    run_btn.click(
        fn=run_and_plot,
        inputs=[mode_dd],
        outputs=[plot_out, summary_out]
    )

    gr.Markdown("""
    ---
    **How to read**: Switch between RL Agent and Random Baseline to see the difference.
    The agent learns to prioritize high-population cities, prevent cascade thresholds,
    and rest teams before fatigue reduces effectiveness.
    """)

demo.launch(share=True)
