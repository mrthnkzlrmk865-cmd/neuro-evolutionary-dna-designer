import numpy as np
import random
import tkinter as tk
import threading
import time
from data_generator import PromoterDatasetGenerator
from neural_network import NeuralNetwork

class NetworkVisualizer(tk.Tk):
    """Tkinter-based GUI to visualize the neural network weights and active DNA sequences."""
    def __init__(self, nn, generator):
        super().__init__()
        self.nn = nn
        self.generator = generator
        self.title("MIT Course 6-7 Portfolio: Synthetic Promoter Evolutionary Visualizer")
        self.geometry("1000x700")
        self.configure(bg='#121212')
        
        self.info_frame = tk.Frame(self, bg='#1e1e1e', bd=2, relief=tk.RIDGE)
        self.info_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.status_label = tk.Label(self.info_frame, text="Status: Initializing...", font=("Courier", 12, "bold"), bg='#1e1e1e', fg='#00FF00')
        self.status_label.pack(side=tk.LEFT, padx=10, pady=5)
        
        self.score_label = tk.Label(self.info_frame, text="Best Score: -", font=("Courier", 12, "bold"), bg='#1e1e1e', fg='#00FFFF')
        self.score_label.pack(side=tk.RIGHT, padx=10, pady=5)

        self.canvas = tk.Canvas(self, bg='#0a0a0a', highlightbackground='#333333')
        self.canvas.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.layer_x = [150, 500, 850]
        self.input_nodes = 15  
        self.hidden_nodes = 12 
        
    def draw_network(self, current_best_seq="-"):
        self.canvas.delete("all")
        
        # Draw W1 Synapses
        for i in range(self.input_nodes):
            for j in range(self.hidden_nodes):
                weight = self.nn.W1[i, j]
                if abs(weight) < 0.25: continue
                
                thickness = min(int(abs(weight) * 2) + 1, 4)
                color = '#00FF00' if weight > 0 else '#FF0000'
                
                y1 = 80 + i * 38
                y2 = 120 + j * 42
                self.canvas.create_line(self.layer_x[0], y1, self.layer_x[1], y2, fill=color, width=thickness)

        # Draw W2 Synapses
        for j in range(self.hidden_nodes):
            weight = self.nn.W2[j, 0]
            thickness = min(int(abs(weight) * 4) + 1, 5)
            color = '#00FF00' if weight > 0 else '#FF0000'
            
            y2 = 120 + j * 42
            y3 = 350
            self.canvas.create_line(self.layer_x[1], y2, self.layer_x[2], y3, fill=color, width=thickness)

        # Draw Input Nodes
        for i in range(self.input_nodes):
            y = 80 + i * 38
            self.canvas.create_oval(self.layer_x[0]-12, y-12, self.layer_x[0]+12, y+12, fill='#333333', outline='#555555')
            if i < len(current_best_seq):
                self.canvas.create_text(self.layer_x[0]-30, y, text=f"b{i}:{current_best_seq[i]}", fill='#FFFFFF', font=("Courier", 10))

        # Draw Hidden Nodes
        for j in range(self.hidden_nodes):
            y = 120 + j * 42
            self.canvas.create_oval(self.layer_x[1]-15, y-15, self.layer_x[1]+15, y+15, fill='#112233', outline='#0088FF', width=2)

        # Draw Output Node
        self.canvas.create_oval(self.layer_x[2]-20, y3-20, self.layer_x[2]+20, y3+20, fill='#221133', outline='#AA00FF', width=3)
        self.canvas.create_text(self.layer_x[2]+50, y3, text="mRNA\nScore", fill='#AA00FF', font=("Courier", 11, "bold"))

    def update_data(self, status, score, best_seq):
        self.status_label.config(text=f"Status: {status}")
        self.score_label.config(text=f"Best Pred: {score:.4f}")
        self.draw_network(best_seq)

    def save_canvas_as_image(self):
        """Saves the current Tkinter canvas state as a high-quality postscript vector image."""
        try:
            self.canvas.update()
            # Saves directly to your project directory
            self.canvas.postscript(file="network_visualization.ps", colormode='color')
            print("\n[SUCCESS] Network visual saved successfully as 'network_visualization.ps'!")
        except Exception as e:
            print(f"\n[ERROR] Could not save canvas image: {e}")

def run_background_evolution(nn, generator, visualizer):
    X, y, _ = generator.create_dataset()
    
    # --- NN Training Phase ---
    for epoch in range(401):
        nn.forward(X)
        nn.backward(y)
        if epoch % 100 == 0:
            loss = np.mean((nn.forward(X) - y) ** 2)
            visualizer.status_label.config(text=f"Training NN... Epoch {epoch}/400")
            visualizer.score_label.config(text=f"Loss: {loss:.5f}")
            visualizer.draw_network()
            time.sleep(0.05)

    # --- Evolutionary Algorithm Phase ---
    pop_size = 40
    mutation_rate = 0.05
    population = ["".join(np.random.choice(generator.bases, generator.seq_len)) for _ in range(pop_size)]
    
    generations = 50
    for gen in range(generations):
        encoded_pop = np.array([generator.one_hot_encode(seq) for seq in population])
        predictions = nn.forward(encoded_pop).flatten()
        
        best_idx = np.argmax(predictions)
        best_score = predictions[best_idx]
        best_seq = population[best_idx]
        
        visualizer.update_data(f"Generation {gen}/{generations}", best_score, best_seq)
        time.sleep(0.1)
        
        total_fit = np.sum(predictions)
        probs = predictions / total_fit if total_fit > 0 else np.ones(pop_size)/pop_size
        selected_indices = np.random.choice(pop_size, size=pop_size, p=probs)
        parents = [population[idx] for idx in selected_indices]
        
        next_pop = []
        for i in range(0, pop_size, 2):
            p1, p2 = parents[i], parents[i+1]
            if random.random() < 0.8:
                pt = random.randint(1, generator.seq_len - 1)
                c1, c2 = p1[:pt] + p2[pt:], p2[:pt] + p1[pt:]
            else:
                c1, c2 = p1, p2
            next_pop.extend([c1, c2])
            
        for i in range(pop_size):
            mutated = list(next_pop[i])
            for m in range(generator.seq_len):
                if random.random() < mutation_rate:
                    mutated[m] = random.choice([b for b in generator.bases if b != mutated[m]])
            next_pop[i] = "".join(mutated)
            
        population = next_pop

    true_score = generator.calculate_expression_score(best_seq)
    visualizer.status_label.config(text="EVOLUTION COMPLETE!")
    visualizer.score_label.config(text=f"Final NN: {best_score:.3f} | Bio-True: {true_score:.3f}")
    
    # Automatically triggers and saves the final snapshot of the canvas
    visualizer.save_canvas_as_image()

if __name__ == "__main__":
    gen = PromoterDatasetGenerator(sequence_length=20, num_samples=1000)
    net = NeuralNetwork(input_size=80, hidden_size=32, output_size=1, learning_rate=0.1)
    
    app = NetworkVisualizer(net, gen)
    
    evo_thread = threading.Thread(target=run_background_evolution, args=(net, gen, app))
    evo_thread.daemon = True
    evo_thread.start()
    
    app.mainloop()
