# Project Setup Journal Entry

## Date: YYYY-MM-DD

### Successful Installations

- **Backtrader**: Installed for versatile backtesting of trading strategies.
- **PyAlgoTrade**: Installed for simplicity in algorithmic trading.
- **QuantLib**: Installed for quantitative finance modeling.

### Mitigations Developed

- **Zipline Compatibility**: Zipline is not compatible with Python 3.12. As a mitigation, alternative backtesting libraries such as Backtrader and PyAlgoTrade were installed that are compatible with Python 3.12.
- **Azure SDK Packages**: Corrected the installation commands for Azure SDK packages. Individual libraries prefixed with `azure-` were installed based on the required services for the project.
- **PyTorch Installation**: Addressed the installation issue by using the correct package name `torch`.

### Notes

- It's important to ensure compatibility of libraries with the Python version in use. In this case, Python 3.12 was the version used, and all libraries installed are compatible with it.
- For any Azure services needed in the future, the corresponding `azure-` prefixed library should be installed.
- Always work within a virtual environment to manage dependencies effectively and avoid conflicts with system-wide installations.

### Next Steps

- Begin developing trading algorithms using the installed backtesting libraries.
- Integrate Azure services as needed for the project.
- Regularly update the journal with progress and any additional installations or mitigations.

---

This journal entry serves as a record of the initial setup process for the project. It will be updated as the project progresses to reflect new developments and changes.
