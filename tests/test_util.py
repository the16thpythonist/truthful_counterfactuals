import os
import pytest

from truthful_counterfactuals.utils import get_version
from truthful_counterfactuals.utils import render_latex
from truthful_counterfactuals.utils import EXPERIMENTS_PATH

from .util import ASSETS_PATH


def test_get_version():
    version = get_version()
    assert isinstance(version, str)
    assert version != ''


@pytest.mark.localonly
def test_render_latex():
    output_path = os.path.join(ASSETS_PATH, 'out.pdf')
    render_latex({'content': '$\pi = 3.141$'}, output_path)
    assert os.path.exists(output_path)
