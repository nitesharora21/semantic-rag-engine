from scripts.compare_retrieval_methods import print_summary_table


def test_print_summary_table_outputs_comparison(capsys) -> None:
    summaries = [
        {
            "method": "Keyword",
            "passed": "2",
            "total": "3",
            "accuracy": "0.67",
        }
    ]
    print_summary_table(summaries=summaries)

    captured = capsys.readouterr()
    assert "Retrieval Method Comparison" in captured.out
    assert "Keyword" in captured.out
    assert "0.67" in captured.out
