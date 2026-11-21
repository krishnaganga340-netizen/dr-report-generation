
import matplotlib.pyplot as plt
import os

# Five prompting strategies
strategies = [
    "Zero-shot",
    "Few-shot",
    "Chain-of-Thought",
    "Template-Constrained",
    "Safety-Constrained"
]

# Evaluation results
data = {
    "Completeness": [4.40, 3.98, 4.14, 4.98, 4.46],
    "Readability": [4.84, 4.98, 4.76, 4.96, 4.74],
    "Consistency": [4.90, 4.00, 4.51, 4.90, 4.48],
    "Medical Safety": [3.30, 5.00, 2.74, 4.68, 4.36],
    "Unsupported Content": [46, 0, 66, 12, 32]
}

# Create a folder to save all graphs
os.makedirs("comparison_charts", exist_ok=True)

# Generate all five graphs
for criterion, scores in data.items():

    plt.figure(figsize=(12, 6))

    bars = plt.bar(strategies, scores)

    plt.title(
        criterion + " Comparison of Prompting Strategies",
        fontsize=15
    )

    if criterion == "Unsupported Content":
        plt.ylabel("Reports with Unsupported Content (%)")
        plt.ylim(0, 100)
    else:
        plt.ylabel("Average Score (Out of 5)")
        plt.ylim(0, 5.5)

    plt.xlabel("Prompting Strategies")
    plt.xticks(rotation=15)

    # Display values above each bar
    for bar, score in zip(bars, scores):
        if criterion == "Unsupported Content":
            label = f"{score}%"
        else:
            label = f"{score:.2f}"

        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.08,
            label,
            ha="center",
            fontsize=10
        )

    plt.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()

    # Save each graph separately
    filename = criterion.lower().replace(" ", "_") + "_comparison.png"

    plt.savefig(
        os.path.join("comparison_charts", filename),
        dpi=300
    )

    plt.close()

print("SUCCESS! All 5 comparison graphs generated.")

