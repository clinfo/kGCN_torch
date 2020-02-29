from ..logger import get_logger


def gcn():
    from .gcn import main
    logger = get_logger("tkgcn")
    main(logger=logger)


def chem():
    from .chem import main
    logger = get_logger("tkgcn-chem")
    main(logger=logger)


def cv_splitter():
    from .cv_splitter import main
    logger = get_logger("tkgcn-cv-splitter-")
    main(logger=logger)


def opt():
    from .opt import main
    logger = get_logger("tkgcn-opt")
    main(logger=logger)


def gen():
    from .gen import main
    logger = get_logger("tkgcn-gen")
    main(logger=logger)


def task_sparse_gcn():
    from .task_sparse_gcn import main
    logger = get_logger("tkgcn-sparse")
    main(logger=logger)
