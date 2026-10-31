import importlib


def test_import_sans_effet_et_fichier_absent(tmp_path, monkeypatch, capsys):
    mod = importlib.import_module("src.model.early_season")   # l'import ne doit rien exécuter
    assert capsys.readouterr().out == ""
    missing = str(tmp_path / "player_logs_2025-26.csv")
    monkeypatch.setattr(mod, "LOGS_PATH", missing)
    assert mod.main() == 1
    err = capsys.readouterr().err
    assert missing in err and "fetch_player_logs" in err and "Traceback" not in err
