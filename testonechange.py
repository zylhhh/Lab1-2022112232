"""
Text Graph Processor Module

This module processes text files to build word graphs, providing various graph analysis
and manipulation functions including bridge word finding, text generation, path
calculation, and visualization.
"""

import argparse
import os
import random
import re
import sys
import time
from collections import defaultdict
from tkinter import (Button, Entry, Frame, Label, Tk, filedialog, messagebox, scrolledtext, simpledialog)

import matplotlib.pyplot as plt
import networkx as nx


class TextGraphProcessor:
    """Processes text to build and analyze word graphs.

    Attributes:
        graph (nx.DiGraph): Directed graph representing word relationships.
        word_list (list): List of processed words from the text.
        original_text (str): The original unprocessed text.
    """

    def __init__(self):
        """Initialize the TextGraphProcessor with empty graph and word list."""
        self.graph = nx.DiGraph()
        self.word_list = []
        self.original_text = ""

    def preprocess_text(self, text):
        """Preprocess text by removing non-alphabetic characters and normalizing.

        Args:
            text (str): Input text to preprocess.

        Returns:
            str: Processed text with only alphabetic characters and single spaces.
        """
        # Use regex to replace all non-alphabetic characters
        text = re.sub(r'[^a-zA-Z\s]', ' ', text)
        # Normalize whitespace and convert to lowercase
        text = re.sub(r'\s+', ' ', text).strip().lower()
        return text

    def build_graph_from_file(self, file_path):
        """Build a word graph from a text file.

        Args:
            file_path (str): Path to the text file.

        Returns:
            bool: True if graph was built successfully, False otherwise.
        """
        try:
            print(f"Reading file: {file_path}")
            start_time = time.time()

            with open(file_path, 'r', encoding='utf-8') as file:
                self.original_text = file.read()

            print(f"File read in {time.time() - start_time:.2f} seconds")

            start_time = time.time()
            processed_text = self.preprocess_text(self.original_text)
            self.word_list = processed_text.split()
            print(f"Text processed in {time.time() - start_time:.2f} seconds")

            # Efficient graph building using edge counting
            start_time = time.time()
            self.graph.clear()
            edge_counts = defaultdict(int)

            # Count all edge occurrences
            for i in range(len(self.word_list) - 1):
                current_word = self.word_list[i]
                next_word = self.word_list[i + 1]
                edge_counts[(current_word, next_word)] += 1

            # Add all edges at once
            for (u, v), weight in edge_counts.items():
                self.graph.add_edge(u, v, weight=weight)

            print(f"Graph built in {time.time() - start_time:.2f} seconds")
            print(f"Graph stats: {len(self.graph)} nodes, "
                  f"{len(self.graph.edges())} edges")

            return True
        except (IOError, UnicodeDecodeError) as e:
            print(f"Error reading file: {e}")
            return False

    def display_graph_text(self):
        """Display the graph representation in text format."""
        if len(self.graph) == 0:
            print("Graph is empty!")
            return

        print("\nGraph representation:")
        print("Nodes:", ", ".join(self.graph.nodes()))
        print("\nEdges with weights:")
        for u, v, data in self.graph.edges(data=True):
            print(f"{u} -> {v} (weight: {data['weight']})")

    def display_graph_visual(self, save_to_file=False, filename="graph.png"):
        """Visualize the graph and optionally save to file.

        Args:
            save_to_file (bool): Whether to save the visualization to a file.
            filename (str): Name of the file to save to.
        """
        if len(self.graph) == 0:
            print("Graph is empty!")
            return

        plt.figure(figsize=(12, 8))

        # Choose layout based on graph size
        if len(self.graph) > 50:
            pos = nx.kamada_kawai_layout(self.graph)
        else:
            pos = nx.spring_layout(self.graph, k=0.5, iterations=50)

        nx.draw_networkx_nodes(
            self.graph, pos, node_size=700, node_color='skyblue')
        nx.draw_networkx_edges(
            self.graph, pos, width=1.5, edge_color='gray', arrows=True)

        # Only show labels for smaller graphs
        if len(self.graph) <= 100:
            nx.draw_networkx_labels(
                self.graph, pos, font_size=10, font_family='sans-serif')

        # Sample edge weights for large graphs
        edge_labels = {}
        for u, v, d in self.graph.edges(data=True):
            if len(self.graph) <= 30 or random.random() < 0.1:
                edge_labels[(u, v)] = d['weight']

        nx.draw_networkx_edge_labels(self.graph, pos, edge_labels=edge_labels)

        plt.title("Text Graph Visualization")
        plt.axis('off')

        if save_to_file:
            plt.savefig(filename)
            print(f"Graph visualization saved to {filename}")
        else:
            plt.show()

    def find_bridge_words(self, word1, word2, max_hops=3):
        """Find bridge words between two words in the graph.

        Args:
            word1 (str): Starting word.
            word2 (str): Target word.
            max_hops (int): Maximum number of hops to consider.

        Returns:
            tuple: (list of bridge words, error message if any)
        """
        word1 = word1.lower()
        word2 = word2.lower()

        if word1 not in self.graph or word2 not in self.graph:
            return None, f"No '{word1}' or '{word2}' in the graph!"

        # Find all possible bridge paths
        bridge_paths = []

        def dfs(current, path, remaining_hops):
            if remaining_hops == 0:
                return
            for neighbor in self.graph.successors(current):
                if neighbor == word2 and len(path) >= 1:
                    bridge_paths.append(path + [neighbor])
                dfs(neighbor, path + [neighbor], remaining_hops - 1)

        dfs(word1, [word1], max_hops)

        if not bridge_paths:
            return [], (f"No bridge words from '{word1}' to '{word2}' "f"within {max_hops} hops!")

        # Extract all bridge words (intermediate nodes in paths)
        bridge_words = set()
        for path in bridge_paths:
            bridge_words.update(path[1:-1])  # Exclude start and end

        return list(bridge_words), None

    def generate_new_text(self, text, max_hops=3):
        """Generate new text by inserting bridge words where possible.

        Args:
            text (str): Input text to process.
            max_hops (int): Maximum hops for bridge word search.

        Returns:
            str: Generated text with bridge words inserted.
        """
        words = self.preprocess_text(text).split()
        if not words:
            return ""

        new_words = [words[0]]

        for i in range(len(words) - 1):
            word1 = words[i]
            word2 = words[i + 1]

            bridge_words, _ = self.find_bridge_words(word1, word2, max_hops)

            if bridge_words:
                chosen_word = random.choice(bridge_words)
                new_words.append(chosen_word)

            new_words.append(word2)

        return ' '.join(new_words)

    def calc_shortest_path(self, word1, word2=None):
        """Calculate shortest path(s) between words.

        Args:
            word1 (str): Starting word.
            word2 (str, optional): Target word. If None, calculates to all nodes.

        Returns:
            tuple: (paths dictionary, lengths dictionary)
        """
        word1 = word1.lower()
        if word2:
            word2 = word2.lower()

        if word1 not in self.graph:
            return None, None

        # Calculate paths to all nodes if no target specified
        if word2 is None:
            paths = {}
            lengths = {}

            for target in self.graph.nodes():
                if target == word1:
                    continue
                try:
                    path = nx.shortest_path(
                        self.graph, source=word1, target=target, weight='weight')
                    length = nx.shortest_path_length(
                        self.graph, source=word1, target=target, weight='weight')
                    paths[target] = path
                    lengths[target] = length
                except nx.NetworkXNoPath:
                    continue

            return paths, lengths

        if word2 not in self.graph:
            return None, None

        try:
            path = nx.shortest_path(
                self.graph, source=word1, target=word2, weight='weight')
            length = nx.shortest_path_length(
                self.graph, source=word1, target=word2, weight='weight')
            return {word2: path}, {word2: length}
        except nx.NetworkXNoPath:
            return None, None

    def calc_all_shortest_paths(self, word1, word2):
        """Calculate all shortest paths between two words.

        Args:
            word1 (str): Starting word.
            word2 (str): Target word.

        Returns:
            tuple: (list of paths, path length)
        """
        word1 = word1.lower()
        word2 = word2.lower()

        if word1 not in self.graph or word2 not in self.graph:
            return None, None

        try:
            paths = list(nx.all_shortest_paths(
                self.graph, source=word1, target=word2, weight='weight'))
            length = nx.shortest_path_length(
                self.graph, source=word1, target=word2, weight='weight')
            return paths, length
        except nx.NetworkXNoPath:
            return None, None

    def calc_pagerank(self, damping_factor=0.85, max_iter=100, tol=1.0e-6):
        """Calculate PageRank for all nodes in the graph.

        Args:
            damping_factor (float): Damping parameter for PageRank.
            max_iter (int): Maximum number of iterations.
            tol (float): Error tolerance for convergence.

        Returns:
            list: Sorted list of (node, score) tuples.
        """
        if len(self.graph) == 0:
            return None

        # Reduce iterations for large graphs
        if len(self.graph) > 1000:
            max_iter = 50

        pr = nx.pagerank(
            self.graph, alpha=damping_factor, max_iter=max_iter, tol=tol)
        return sorted(pr.items(), key=lambda x: x[1], reverse=True)

    def random_walk(self):
        """Perform a random walk on the graph.

        Returns:
            str: The path taken during the random walk.
        """
        if len(self.graph) == 0:
            return None

        visited_edges = set()
        path = []

        # Random starting node
        current_node = random.choice(list(self.graph.nodes()))
        path.append(current_node)

        while True:
            out_edges = list(self.graph.out_edges(current_node))
            if not out_edges:
                break

            # Randomly choose an outgoing edge
            chosen_edge = random.choice(out_edges)

            # Stop if we've visited this edge before
            if chosen_edge in visited_edges:
                path.append(chosen_edge[1])  # Add target node
                break

            visited_edges.add(chosen_edge)
            next_node = chosen_edge[1]
            path.append(next_node)
            current_node = next_node

        return ' '.join(path)

    def save_walk_to_file(self, walk_text, filename="random_walk.txt"):
        """Save random walk result to a file.

        Args:
            walk_text (str): Text to save.
            filename (str): Name of the file to save to.

        Returns:
            bool: True if save was successful, False otherwise.
        """
        try:
            with open(filename, 'w', encoding='utf-8') as file:
                file.write(walk_text)
            return True
        except (IOError, OSError) as e:
            print(f"Error saving walk to file: {e}")
            return False


class CommandLineInterface:
    """Command line interface for the Text Graph Processor."""

    def __init__(self):
        """Initialize the CLI with a TextGraphProcessor instance."""
        self.processor = TextGraphProcessor()

    def run(self):
        """Run the command line interface."""
        print("Text Graph Processor - Command Line Interface")
        print("-------------------------------------------")

        # Get file path from arguments or input
        if len(sys.argv) > 1 and sys.argv[1] == '--file':
            file_path = sys.argv[2] if len(sys.argv) > 2 else None
        else:
            file_path = input("Enter the path to the text file: ").strip()

        if not file_path or not os.path.exists(file_path):
            print("File does not exist!")
            return

        # Build the graph
        if not self.processor.build_graph_from_file(file_path):
            print("Failed to build graph from file.")
            return

        print(f"Graph built successfully with {len(self.processor.graph)} "
              f"nodes and {len(self.processor.graph.edges())} edges.")

        while True:
            print("\nMenu:")
            print("1. Display graph")
            print("2. Find bridge words")
            print("3. Generate new text with bridge words")
            print("4. Calculate shortest path from one word to another")
            print("5. Calculate all shortest paths from one word to all others")
            print("6. Calculate PageRank")
            print("7. Random walk")
            print("8. Exit")

            choice = input("Enter your choice (1-8): ").strip()

            if choice == '1':
                self._handle_display_graph()
            elif choice == '2':
                self._handle_find_bridge_words()
            elif choice == '3':
                self._handle_generate_text()
            elif choice == '4':
                self._handle_shortest_path()
            elif choice == '5':
                self._handle_all_shortest_paths()
            elif choice == '6':
                self._handle_pagerank()
            elif choice == '7':
                self._handle_random_walk()
            elif choice == '8':
                print("Exiting...")
                break
            else:
                print("Invalid choice. Please try again.")

    def _handle_display_graph(self):
        """Handle the display graph menu option."""
        self.processor.display_graph_text()

        viz_choice = input("\nShow visual graph representation? (y/n): ").strip().lower()
        if viz_choice == 'y':
            save_choice = input("Save graph to file? (y/n): ").strip().lower()
            if save_choice == 'y':
                filename = input("Enter filename (default: graph.png): ").strip() or "graph.png"
                self.processor.display_graph_visual(save_to_file=True, filename=filename)
            else:
                self.processor.display_graph_visual()

    def _handle_find_bridge_words(self):
        """Handle the find bridge words menu option."""
        word1 = input("Enter first word: ").strip()
        word2 = input("Enter second word: ").strip()
        max_hops = int(input("Enter max hops (default 3): ") or 3)

        bridge_words, error_msg = self.processor.find_bridge_words(word1, word2, max_hops)

        if error_msg:
            print(error_msg)
        elif not bridge_words:
            print(f"No bridge words from '{word1}' to '{word2}' within {max_hops} hops!")
        else:
            if len(bridge_words) == 1:
                print(f"The bridge word from '{word1}' to '{word2}' is: {bridge_words[0]}")
            else:
                bridge_words_str = ', '.join(bridge_words[:-1]) + f" and {bridge_words[-1]}"
                print(f"The bridge words from '{word1}' to '{word2}' are: {bridge_words_str}")

    def _handle_generate_text(self):
        """Handle the generate text menu option."""
        text = input("Enter new text: ").strip()
        new_text = self.processor.generate_new_text(text)
        print("\nGenerated text:", new_text)

    def _handle_shortest_path(self):
        """Handle the shortest path menu option."""
        word1 = input("Enter first word: ").strip()
        word2 = input("Enter second word (leave blank to show all paths from first word): ").strip()

        if word2:
            paths, lengths = self.processor.calc_shortest_path(word1, word2)
            if paths is None:
                print(f"No path from {word1} to {word2}!")
            else:
                path = paths[word2]
                length = lengths[word2]
                print(f"\nShortest path from {word1} to {word2}: {' -> '.join(path)}")
                print(f"Path length: {length}")
        else:
            print(f"\nCalculating all shortest paths from '{word1}'...")
            paths, lengths = self.processor.calc_shortest_path(word1)

            if not paths:
                print(f"No paths found from {word1}!")
            else:
                print(f"Shortest paths from {word1}:")
                for target, path in sorted(paths.items(), key=lambda x: len(x[1])):
                    length = lengths[target]
                    print(f"To {target}: {' -> '.join(path)} (length: {length})")

    def _handle_all_shortest_paths(self):
        """Handle the all shortest paths menu option."""
        word1 = input("Enter word to calculate all shortest paths from: ").strip()
        print(f"\nCalculating all shortest paths from '{word1}'...")

        paths, lengths = self.processor.calc_shortest_path(word1)

        if not paths:
            print(f"No paths found from {word1}!")
        else:
            print(f"\nAll shortest paths from {word1}:")
            for target, path in sorted(paths.items(), key=lambda x: lengths[x[0]]):
                length = lengths[target]
                print(f"To {target}: {' -> '.join(path)} (length: {length})")

    def _handle_pagerank(self):
        """Handle the PageRank menu option."""
        print("\nCalculating PageRank...")
        pr_results = self.processor.calc_pagerank()
        if pr_results:
            print("\nPageRank results (top 20):")
            for word, score in pr_results[:20]:
                print(f"{word}: {score:.4f}")

    def _handle_random_walk(self):
        """Handle the random walk menu option."""
        print("\nPerforming random walk...")
        walk_text = self.processor.random_walk()
        print("\nRandom walk result:", walk_text)

        save_choice = input("\nSave walk to file? (y/n): ").strip().lower()
        if save_choice == 'y':
            filename = input("Enter filename (default: random_walk.txt): ").strip() or "random_walk.txt"
            if self.processor.save_walk_to_file(walk_text, filename):
                print(f"Walk saved to {filename}")
            else:
                print("Failed to save walk.")


class GraphicalUserInterface:
    """Graphical user interface for the Text Graph Processor."""

    def __init__(self, root):
        """Initialize the GUI.

        Args:
            root (tk.Tk): The root Tkinter window.
        """
        self.root = root
        self.root.title("Text Graph Processor")
        self.processor = TextGraphProcessor()

        # Main frame
        self.main_frame = Frame(root)
        self.main_frame.pack(padx=10, pady=10, fill='both', expand=True)

        # File selection section
        self._setup_file_frame()

        # Result display area
        self.result_text = scrolledtext.ScrolledText(
            self.main_frame, width=80, height=20)
        self.result_text.pack(fill='both', expand=True, pady=5)

        # Function buttons
        self._setup_button_frame()

    def _setup_file_frame(self):
        """Set up the file selection frame."""
        self.file_frame = Frame(self.main_frame)
        self.file_frame.pack(fill='x', pady=5)

        self.file_label = Label(self.file_frame, text="Text File:")
        self.file_label.pack(side='left')

        self.file_entry = Entry(self.file_frame, width=50)
        self.file_entry.pack(side='left', padx=5, expand=True, fill='x')

        self.browse_button = Button(
            self.file_frame, text="Browse", command=self.browse_file)
        self.browse_button.pack(side='left')

        self.load_button = Button(
            self.file_frame, text="Load", command=self.load_file)
        self.load_button.pack(side='left', padx=5)

    def _setup_button_frame(self):
        """Set up the function buttons frame."""
        self.button_frame = Frame(self.main_frame)
        self.button_frame.pack(fill='x', pady=5)

        buttons = [
            ("Show Graph", self.show_graph),
            ("Find Bridge Words", self.find_bridge_words),
            ("Generate Text", self.generate_text),
            ("Shortest Path", self.shortest_path),
            ("All Shortest Paths", self.all_shortest_paths),
            ("PageRank", self.show_pagerank),
            ("Random Walk", self.random_walk)
        ]

        for text, command in buttons:
            Button(
                self.button_frame, text=text, command=command
            ).pack(side='left', padx=2)

    def browse_file(self):
        """Open file dialog to select a text file."""
        file_path = filedialog.askopenfilename(
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")])
        if file_path:
            self.file_entry.delete(0, 'end')
            self.file_entry.insert(0, file_path)

    def load_file(self):
        """Load the selected file and build the graph."""
        file_path = self.file_entry.get()
        if not file_path:
            messagebox.showerror("Error", "Please select a file first.")
            return

        if not os.path.exists(file_path):
            messagebox.showerror("Error", "File does not exist!")
            return

        if self.processor.build_graph_from_file(file_path):
            self.result_text.insert(
                'end', f"Graph built successfully with {len(self.processor.graph)} "
                f"nodes and {len(self.processor.graph.edges())} edges.\n")
        else:
            messagebox.showerror("Error", "Failed to build graph from file.")

    def show_graph(self):
        """Display the graph in text and optionally visual form."""
        if len(self.processor.graph) == 0:
            messagebox.showerror("Error", "Graph is empty. Please load a file first.")
            return

        # Display text representation
        self.result_text.insert('end', "\nGraph representation:\n")
        self.result_text.insert('end', "Nodes: " + ", ".join(self.processor.graph.nodes()) + "\n\n")
        self.result_text.insert('end', "Edges with weights:\n")
        for u, v, data in self.processor.graph.edges(data=True):
            self.result_text.insert('end', f"{u} -> {v} (weight: {data['weight']})\n")

        # Ask about visualization
        if messagebox.askyesno("Visualization", "Show visual graph representation?"):
            save_choice = messagebox.askyesno("Save Graph", "Save graph to file?")
            if save_choice:
                filename = filedialog.asksaveasfilename(
                    defaultextension=".png", filetypes=[("PNG files", "*.png")])
                if filename:
                    self.processor.display_graph_visual(save_to_file=True, filename=filename)
                    self.result_text.insert('end', f"Graph visualization saved to {filename}\n")
            else:
                self.processor.display_graph_visual()

    def find_bridge_words(self):
        """Find and display bridge words between two words."""
        if len(self.processor.graph) == 0:
            messagebox.showerror("Error", "Graph is empty. Please load a file first.")
            return

        word1 = simpledialog.askstring("Input", "Enter first word:")
        if not word1:
            return

        word2 = simpledialog.askstring("Input", "Enter second word:")
        if not word2:
            return

        bridge_words = self.processor.find_bridge_words(word1, word2)

        if bridge_words is None:
            self.result_text.insert('end', f"No {word1} or {word2} in the graph!\n")
        elif not bridge_words:
            self.result_text.insert('end', f"No bridge words from {word1} to {word2}!\n")
        else:
            if len(bridge_words) == 1:
                self.result_text.insert('end', f"The bridge word from {word1} to {word2} is: {bridge_words[0]}\n")
            else:
                self.result_text.insert('end', f"The bridge words from {word1} to {word2} are: {', '.join(bridge_words[:-1])} and {bridge_words[-1]}\n")

    def generate_text(self):
        """Generate and display new text with bridge words."""
        if len(self.processor.graph) == 0:
            messagebox.showerror("Error", "Graph is empty. Please load a file first.")
            return

        text = simpledialog.askstring("Input", "Enter new text:")
        if not text:
            return

        new_text = self.processor.generate_new_text(text)
        self.result_text.insert('end', f"\nOriginal text: {text}\n")
        self.result_text.insert('end', f"Generated text: {new_text}\n")

    def shortest_path(self):
        """Calculate and display shortest path(s)."""
        if len(self.processor.graph) == 0:
            messagebox.showerror("Error", "Graph is empty. Please load a file first.")
            return

        word1 = simpledialog.askstring("Input", "Enter first word:")
        if not word1:
            return

        word2 = simpledialog.askstring("Input", "Enter second word (leave blank for all paths):")

        if word2:
            paths, lengths = self.processor.calc_shortest_path(word1, word2)
            if paths is None:
                self.result_text.insert('end', f"No path from {word1} to {word2}!\n")
            else:
                path = paths[word2]
                length = lengths[word2]
                self.result_text.insert('end', f"\nShortest path from {word1} to {word2}: {' -> '.join(path)}\n")
                self.result_text.insert('end', f"Path length: {length}\n")
        else:
            self.result_text.insert('end', f"\nCalculating all shortest paths from '{word1}'...\n")
            paths, lengths = self.processor.calc_shortest_path(word1)

            if not paths:
                self.result_text.insert('end', f"No paths found from {word1}!\n")
            else:
                self.result_text.insert('end', f"Shortest paths from {word1}:\n")
                for target, path in sorted(paths.items(), key=lambda x: lengths[x[0]]):
                    length = lengths[target]
                    self.result_text.insert('end', f"To {target}: {' -> '.join(path)} (length: {length})\n")

    def all_shortest_paths(self):
        """Calculate and display all shortest paths from a word."""
        if len(self.processor.graph) == 0:
            messagebox.showerror("Error", "Graph is empty. Please load a file first.")
            return

        word1 = simpledialog.askstring("Input", "Enter word to calculate all shortest paths from:")
        if not word1:
            return

        self.result_text.insert('end', f"\nCalculating all shortest paths from '{word1}'...\n")
        paths, lengths = self.processor.calc_shortest_path(word1)

        if not paths:
            self.result_text.insert('end', f"No paths found from {word1}!\n")
        else:
            self.result_text.insert('end', f"\nAll shortest paths from {word1}:\n")
            for target, path in sorted(paths.items(), key=lambda x: lengths[x[0]]):
                length = lengths[target]
                self.result_text.insert('end', f"To {target}: {' -> '.join(path)} (length: {length})\n")

    def show_pagerank(self):
        """Calculate and display PageRank results."""
        if len(self.processor.graph) == 0:
            messagebox.showerror("Error", "Graph is empty. Please load a file first.")
            return

        self.result_text.insert('end', "\nCalculating PageRank...\n")
        pr_results = self.processor.calc_pagerank()
        if pr_results:
            self.result_text.insert('end', "\nPageRank results (top 20):\n")
            for word, score in pr_results[:20]:
                self.result_text.insert('end', f"{word}: {score:.4f}\n")

    def random_walk(self):
        """Perform and display a random walk."""
        if len(self.processor.graph) == 0:
            messagebox.showerror("Error", "Graph is empty. Please load a file first.")
            return

        self.result_text.insert('end', "\nPerforming random walk...\n")
        walk_text = self.processor.random_walk()
        self.result_text.insert('end', f"\nRandom walk result: {walk_text}\n")

        if messagebox.askyesno("Save Walk", "Save walk to file?"):
            filename = filedialog.asksaveasfilename(
                defaultextension=".txt", filetypes=[("Text files", "*.txt")])
            if filename:
                if self.processor.save_walk_to_file(walk_text, filename):
                    self.result_text.insert('end', f"Walk saved to {filename}\n")
                else:
                    self.result_text.insert('end', "Failed to save walk.\n")


def main():
    """Main function to start the application."""
    parser = argparse.ArgumentParser(description="Text Graph Processor")
    parser.add_argument('--file', type=str, help="Path to the text file")
    parser.add_argument('--gui', action='store_true', help="Use graphical user interface")
    args = parser.parse_args()

    if args.gui:
        root = Tk()
        GraphicalUserInterface(root)
        root.mainloop()
    else:
        cli = CommandLineInterface()
        cli.run()


if __name__ == "__main__":
    main()
