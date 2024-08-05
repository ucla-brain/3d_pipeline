
def pytest_addoption(parser):
    parser.addoption(
        "--no_overwrite", action="store_true", default=False, help="Skip generating obj files for locations with output folders containing obj(s) files"
    )
    parser.addoption(
        "--verbose_log", action="store_true", default=False, help="Print verbose log messages"        
    )    