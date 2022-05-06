from click.testing import CliRunner
from bot_spotter.cli import cli


class TestCli:
    def test_run(self):
        args = [
            "--input-file-path",
            "./data/fake_users.csv",
            "--output-dir",
            "./artifact",
            "--val-mode",
        ]

        runner = CliRunner()
        result = runner.invoke(cli.run, args=args)
        expected_input_file_path = "./data/fake_users.csv"
        expected_output_dir = "./artifact"
        expected_val_mode = True
        assert (
            result.output
            == f"input path of data: {expected_input_file_path}\noutput path of artifacts: {expected_output_dir}\n"
            f"training mode: {expected_val_mode}\n"
        )
