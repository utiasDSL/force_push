# mm2d

Simulation tools for a two-dimensional mobile manipulator. You'll need
[pipenv](https://pipenv.pypa.io) to manage the virtual environment and
dependencies.

## Install

```bash
# Clone the repository
git clone https://github.com/adamheins/mm2d

# Install basic dependencies
cd mm2d
pipenv install

# Activate the virtualenv
pipenv shell

# Install the mm2d package for development
python setup.py develop

# Manually install qpOASES (it's not available from pip) and its Python bindings
git clone https://github.com/coin-or/qpOASES /path/to/qpOASES
cd /path/to/qpOASES
make
cd interfaces/python
python setup.py install
```

## Organization

* `mm2d`: The main Python package.
* `sims`: Simulations making use of the `mm2d` utilities.
* `tools`: Miscellaneous additional utilities.

## License

MIT - see the LICENSE file.
