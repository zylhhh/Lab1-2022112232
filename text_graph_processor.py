import re
import networkx as nx
import matplotlib.pyplot as plt
import random
import argparse
from tkinter import Tk, filedialog, simpledialog, messagebox, scrolledtext, Button, Entry, Label, Frame, StringVar, OptionMenu
from collections import defaultdict
import os
import sys
import time

class TextGraphProcessor:
    def __init__(self):
        self.graph = nx.DiGraph()
        self.word_list = []
        self.original_text = ""
        
    def preprocess_text(self, text):
        # 使用正则表达式一次性替换所有非字母字符
        text = re.sub(r'[^a-zA-Z\s]', ' ', text)
        # 将多个空格合并为一个并转换为小写
        text = re.sub(r'\s+', ' ', text).strip().lower()
        return text
    
    def build_graph_from_file(self, file_path):
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
            
            # 使用更高效的方式构建图
            start_time = time.time()
            self.graph.clear()
            edge_counts = defaultdict(int)
            
            # 先统计所有边及其出现次数
            for i in range(len(self.word_list) - 1):
                current_word = self.word_list[i]
                next_word = self.word_list[i+1]
                edge_counts[(current_word, next_word)] += 1
            
            # 一次性添加所有边
            for (u, v), weight in edge_counts.items():
                self.graph.add_edge(u, v, weight=weight)
            
            print(f"Graph built in {time.time() - start_time:.2f} seconds")
            print(f"Graph stats: {len(self.graph)} nodes, {len(self.graph.edges())} edges")
            
            return True
        except Exception as e:
            print(f"Error reading file: {e}")
            return False
    
    def display_graph_text(self):
        """在命令行中展示图的文本表示"""
        if len(self.graph) == 0:
            print("Graph is empty!")
            return
        
        print("\nGraph representation:")
        print("Nodes:", ", ".join(self.graph.nodes()))
        print("\nEdges with weights:")
        for u, v, data in self.graph.edges(data=True):
            print(f"{u} -> {v} (weight: {data['weight']})")
    
    def display_graph_visual(self, save_to_file=False, filename="graph.png"):
        """可视化图并可选保存到文件"""
        if len(self.graph) == 0:
            print("Graph is empty!")
            return
        
        plt.figure(figsize=(12, 8))
        
        # 对于大图，使用不同的布局算法
        if len(self.graph) > 50:
            pos = nx.kamada_kawai_layout(self.graph)
        else:
            pos = nx.spring_layout(self.graph, k=0.5, iterations=50)
        
        nx.draw_networkx_nodes(self.graph, pos, node_size=700, node_color='skyblue')
        nx.draw_networkx_edges(self.graph, pos, width=1.5, edge_color='gray', arrows=True)
        
        # 对于大图，不显示所有节点标签
        if len(self.graph) <= 100:
            nx.draw_networkx_labels(self.graph, pos, font_size=10, font_family='sans-serif')
        
        # 只显示部分边的权重
        edge_labels = {}
        for u, v, d in self.graph.edges(data=True):
            if len(self.graph) <= 30 or random.random() < 0.1:  # 对小图显示全部，大图抽样显示
                edge_labels[(u, v)] = d['weight']
        
        nx.draw_networkx_edge_labels(self.graph, pos, edge_labels=edge_labels)
        
        plt.title("Text Graph Visualization")
        plt.axis('off')
        
        if save_to_file:
            plt.savefig(filename)
            print(f"Graph visualization saved to {filename}")
        else:
            plt.show()
    
    def find_bridge_words(self, word1, word2):
        word1 = word1.lower()
        word2 = word2.lower()
        
        if word1 not in self.graph or word2 not in self.graph:
            return None
        
        # 使用集合交集提高效率
        successors = set(self.graph.successors(word1))
        predecessors = set(self.graph.predecessors(word2))
        bridge_words = list(successors & predecessors)
        
        return bridge_words if bridge_words else None
    
    def generate_new_text(self, text):
        words = self.preprocess_text(text).split()
        if not words:
            return ""
            
        new_words = [words[0]]  # 从第一个词开始
        
        for i in range(len(words) - 1):
            word1 = words[i]
            word2 = words[i+1]
            
            # 检查是否有桥接词
            bridge_words = self.find_bridge_words(word1, word2)
            if bridge_words:
                chosen_word = random.choice(bridge_words)
                new_words.append(chosen_word)
            
            new_words.append(word2)
        
        return ' '.join(new_words)
    
    def calc_shortest_path(self, word1, word2=None):
        word1 = word1.lower()
        if word2:
            word2 = word2.lower()
        
        if word1 not in self.graph:
            return None, None
        
        # 如果只提供了一个单词，计算到所有其他节点的最短路径
        if word2 is None:
            paths = {}
            lengths = {}
            
            # 使用单源最短路径算法提高效率
            for target in self.graph.nodes():
                if target == word1:
                    continue
                try:
                    path = nx.shortest_path(self.graph, source=word1, target=target, weight='weight')
                    length = nx.shortest_path_length(self.graph, source=word1, target=target, weight='weight')
                    paths[target] = path
                    lengths[target] = length
                except nx.NetworkXNoPath:
                    continue
            
            return paths, lengths
        else:
            if word2 not in self.graph:
                return None, None
            
            try:
                path = nx.shortest_path(self.graph, source=word1, target=word2, weight='weight')
                length = nx.shortest_path_length(self.graph, source=word1, target=word2, weight='weight')
                return {word2: path}, {word2: length}
            except nx.NetworkXNoPath:
                return None, None
    
    def calc_all_shortest_paths(self, word1, word2):
        word1 = word1.lower()
        word2 = word2.lower()
        
        if word1 not in self.graph or word2 not in self.graph:
            return None, None
        
        try:
            paths = list(nx.all_shortest_paths(self.graph, source=word1, target=word2, weight='weight'))
            length = nx.shortest_path_length(self.graph, source=word1, target=word2, weight='weight')
            return paths, length
        except nx.NetworkXNoPath:
            return None, None
    
    def calc_pagerank(self, damping_factor=0.85, max_iter=100, tol=1.0e-6):
        if len(self.graph) == 0:
            return None
        
        # 对于大图，减少迭代次数以加快计算
        if len(self.graph) > 1000:
            max_iter = 50
        
        pr = nx.pagerank(self.graph, alpha=damping_factor, max_iter=max_iter, tol=tol)
        return sorted(pr.items(), key=lambda x: x[1], reverse=True)
    
    def random_walk(self):
        if len(self.graph) == 0:
            return None
        
        visited_edges = set()
        path = []
        
        # 随机选择起始节点
        current_node = random.choice(list(self.graph.nodes()))
        path.append(current_node)
        
        while True:
            # 获取当前节点的所有出边
            out_edges = list(self.graph.out_edges(current_node))
            if not out_edges:
                break
            
            # 随机选择一条出边
            chosen_edge = random.choice(out_edges)
            
            # 检查是否已经访问过这条边
            if chosen_edge in visited_edges:
                path.append(chosen_edge[1])  # 添加目标节点
                break
            
            visited_edges.add(chosen_edge)
            next_node = chosen_edge[1]
            path.append(next_node)
            current_node = next_node
        
        return ' '.join(path)
    
    def save_walk_to_file(self, walk_text, filename="random_walk.txt"):
        try:
            with open(filename, 'w', encoding='utf-8') as file:
                file.write(walk_text)
            return True
        except Exception as e:
            print(f"Error saving walk to file: {e}")
            return False

class CommandLineInterface:
    def __init__(self):
        self.processor = TextGraphProcessor()
    
    def run(self):
        print("Text Graph Processor - Command Line Interface")
        print("-------------------------------------------")
        
        # 获取文件路径
        if len(sys.argv) > 1 and sys.argv[1] == '--file':
            file_path = sys.argv[2] if len(sys.argv) > 2 else None
        else:
            file_path = input("Enter the path to the text file: ").strip()
        
        if not file_path or not os.path.exists(file_path):
            print("File does not exist!")
            return
        
        # 构建图
        if not self.processor.build_graph_from_file(file_path):
            print("Failed to build graph from file.")
            return
        
        print(f"Graph built successfully with {len(self.processor.graph)} nodes and {len(self.processor.graph.edges())} edges.")
        
        while True:
            print("\nMenu:")
            print("1. 展示图")
            print("2. 寻找桥接词")
            print("3. 生成新文本")
            print("4. 计算最短距离")
            print("5. 计算一个词到所有词最短距离")
            print("6. 计算PageRank")
            print("7. 随机游走")
            print("8. Exit")
            
            choice = input("Enter your choice (1-8): ").strip()
            
            if choice == '1':
                # 先在命令行中展示图的文本表示
                self.processor.display_graph_text()
                
                # 然后询问是否要可视化
                viz_choice = input("\nShow visual graph representation? (y/n): ").strip().lower()
                if viz_choice == 'y':
                    save_choice = input("Save graph to file? (y/n): ").strip().lower()
                    if save_choice == 'y':
                        filename = input("Enter filename (default: graph.png): ").strip() or "graph.png"
                        self.processor.display_graph_visual(save_to_file=True, filename=filename)
                    else:
                        self.processor.display_graph_visual()
            
            elif choice == '2':
                word1 = input("Enter first word: ").strip()
                word2 = input("Enter second word: ").strip()
                bridge_words = self.processor.find_bridge_words(word1, word2)
                
                if bridge_words is None:
                    print(f"No {word1} or {word2} in the graph!")
                elif not bridge_words:
                    print(f"No bridge words from {word1} to {word2}!")
                else:
                    if len(bridge_words) == 1:
                        print(f"The bridge word from {word1} to {word2} is: {bridge_words[0]}")
                    else:
                        print(f"The bridge words from {word1} to {word2} are: {', '.join(bridge_words[:-1])} and {bridge_words[-1]}")
            
            elif choice == '3':
                text = input("Enter new text: ").strip()
                new_text = self.processor.generate_new_text(text)
                print("\nGenerated text:", new_text)
            
            elif choice == '4':
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
            
            elif choice == '5':
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
            
            elif choice == '6':
                print("\nCalculating PageRank...")
                pr_results = self.processor.calc_pagerank()
                if pr_results:
                    print("\nPageRank results (top 20):")
                    for word, score in pr_results[:20]:
                        print(f"{word}: {score:.4f}")
            
            elif choice == '7':
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
            
            elif choice == '8':
                print("Exiting...")
                break
            
            else:
                print("Invalid choice. Please try again.")

def main():
    parser = argparse.ArgumentParser(description="Text Graph Processor")
    parser.add_argument('--file', type=str, help="Path to the text file")
    parser.add_argument('--gui', action='store_true', help="Use graphical user interface")
    args = parser.parse_args()
    
    if args.gui:
        root = Tk()
        app = GraphicalUserInterface(root)
        root.mainloop()
    else:
        cli = CommandLineInterface()
        cli.run()

if __name__ == "__main__":
    main()