Installation
============

Since Hyperion (v1.0.5) is currently in development and not yet available via pip on PyPI, it must be installed manually from the source code.

Follow these steps to install the library on your system:

1. **Clone the repository**

   .. code-block:: bash

      git clone <repository-url>

2. **Navigate to the hyperion directory**

   .. code-block:: bash

      cd hyperion

3. **Install dependencies**

   Install the required packages listed in the ``requirements.txt`` file:

   .. code-block:: bash

      pip install -r requirements.txt

4. **Install the package**

   You can install the library into your Python environment using the ``setup.py`` script:

   .. code-block:: bash

      python setup.py install

   Alternatively, if you prefer not to install it but want to use it directly from the directory, you can add it to your ``PYTHONPATH``:

   .. code-block:: bash

      export PYTHONPATH="/path/to/hyperion:$PYTHONPATH"

5. **Verify installation**

   To ensure that Hyperion has been installed correctly, open a Python interpreter and verify the installation:

   .. code-block:: python

      import hyperion
      print(hyperion.__version__)

Once the setup is successful, refer to the :doc:`user_guide` to start using the library.
