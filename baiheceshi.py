import pytest
from text_graph_processor import TextGraphProcessor

@pytest.fixture
def text_graph_processor():
    processor = TextGraphProcessor()
    # 使用Easy Test.txt的内容构建图
    test_text = """The scientist carefully analyzed the data, wrote a detailed report, and shared the report with the team, but the team requested more data, so the scientist analyzed it again."""
    processor.original_text = test_text
    processed_text = processor.preprocess_text(test_text)
    processor.word_list = processed_text.split()
    
    # 构建图
    edge_counts = {}
    for i in range(len(processor.word_list) - 1):
        current_word = processor.word_list[i]
        next_word = processor.word_list[i+1]
        edge_key = (current_word, next_word)
        edge_counts[edge_key] = edge_counts.get(edge_key, 0) + 1
    
    for (u, v), weight in edge_counts.items():
        processor.graph.add_edge(u, v, weight=weight)
    
    return processor

def test_shortest_path_nonexistent_words(text_graph_processor):
    # 测试用例1: word1="if" word2="data" → (None, None)
    paths, lengths = text_graph_processor.calc_shortest_path("if", "data")
    assert paths is None
    assert lengths is None

def test_shortest_path_from_scientist_to_all(text_graph_processor):
    # 测试用例2: word1="scientist" word2=None → 包含所有目标节点的路径和长度字典
    paths, lengths = text_graph_processor.calc_shortest_path("scientist")
    
    assert isinstance(paths, dict)
    assert isinstance(lengths, dict)
    assert len(paths) > 0
    assert len(lengths) > 0
    
    # 检查一些预期的路径是否存在
    assert "data" in paths
    assert "team" in paths
    assert "report" in paths
    
    # 检查路径长度是否正确
    assert lengths["data"] == 3  # scientist -> carefully -> analyzed -> the -> data
    assert lengths["team"] == 4  # scientist -> carefully -> analyzed -> the -> data -> wrote -> a -> detailed -> report -> and -> shared -> the -> report -> with -> the -> team

def test_shortest_path_from_data_to_all(text_graph_processor):
    # 测试用例3: word1="data" word2=None → 包含部分目标节点的路径和长度字典
    paths, lengths = text_graph_processor.calc_shortest_path("it")
    
    assert isinstance(paths, dict)
    assert isinstance(lengths, dict)
    assert len(paths) > 0
    
    # 检查一些预期的路径是否存在
    assert "again" in paths
    
    assert "carefully" not in paths 

def test_shortest_path_scientist_to_data(text_graph_processor):
    # 测试用例4: word1="scientist" word2="data" → 返回正确的路径和长度
    paths, lengths = text_graph_processor.calc_shortest_path("scientist", "data")
    
    assert isinstance(paths, dict)
    assert isinstance(lengths, dict)
    assert "data" in paths
    assert "data" in lengths
    
    # 检查路径是否正确
    expected_path = ['scientist','analyzed','the','data']
    assert paths["data"] == expected_path
    
    # 检查路径长度是否正确
    assert lengths["data"] == 3  # 边的数量是4 (节点数-1)

def test_shortest_path_team_to_analyzed(text_graph_processor):
    # 测试用例5: word1="team" word2="analyzed" → (None, None)
    paths, lengths = text_graph_processor.calc_shortest_path("again", "the")
    assert paths is None
    assert lengths is None

def test_shortest_path_scientist_to_if(text_graph_processor):
    # 测试用例6: word1="scientist" word2="if" → (None, None)
    paths, lengths = text_graph_processor.calc_shortest_path("scientist", "if")
    assert paths is None
    assert lengths is None