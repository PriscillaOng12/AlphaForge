# Contributing to Factor Model Backtester

We welcome contributions to the Factor Model Backtester! This document provides guidelines for contributing to the project.

## 🤝 How to Contribute

### Reporting Issues

1. **Search existing issues** first to avoid duplicates
2. **Use the issue template** when creating new issues
3. **Provide detailed information**:
   - Clear description of the problem
   - Steps to reproduce
   - Expected vs actual behavior
   - Environment details (Python version, OS, etc.)
   - Code samples if applicable

### Suggesting Features

1. **Check the roadmap** to see if the feature is already planned
2. **Open a feature request** with detailed description
3. **Explain the use case** and why it would be valuable
4. **Consider implementation complexity** and backwards compatibility

### Code Contributions

1. **Fork the repository**
2. **Create a feature branch** (`git checkout -b feature/amazing-feature`)
3. **Make your changes** following our coding standards
4. **Add tests** for new functionality
5. **Update documentation** as needed
6. **Submit a pull request**

## 🔧 Development Setup

### Prerequisites

- Python 3.7+ 
- Git
- Virtual environment tool (venv, conda, etc.)

### Local Development

```bash
# Clone your fork
git clone https://github.com/yourusername/factor-model-backtester.git
cd factor-model-backtester

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install development dependencies
pip install pytest pytest-cov black isort flake8 mypy

# Install package in development mode
pip install -e .
```

### Running Tests

```bash
# Run all tests
python -m pytest

# Run with coverage
python -m pytest --cov=factor_backtester

# Run specific test file
python -m pytest tests/test_factor_backtester.py

# Run tests with verbose output
python -m pytest -v
```

### Code Quality

```bash
# Format code with black
black factor_backtester.py tests/

# Sort imports with isort
isort factor_backtester.py tests/

# Check code style with flake8
flake8 factor_backtester.py tests/

# Type checking with mypy
mypy factor_backtester.py
```

## 📝 Coding Standards

### Python Style

- Follow **PEP 8** style guide
- Use **type hints** for function parameters and return values
- Write **docstrings** for all public functions and classes
- Keep functions **focused and small** (< 50 lines when possible)
- Use **meaningful variable names**

### Example Code Style

```python
def calculate_factor_score(
    prices: pd.DataFrame, 
    lookback_window: int = 252,
    min_observations: int = 100
) -> pd.Series:
    """
    Calculate factor score based on price momentum.
    
    Args:
        prices: DataFrame with price data
        lookback_window: Number of days for calculation
        min_observations: Minimum required observations
        
    Returns:
        Series with factor scores
        
    Raises:
        ValueError: If insufficient data provided
    """
    if len(prices) < min_observations:
        raise ValueError(f"Insufficient data: {len(prices)} < {min_observations}")
    
    # Implementation here
    return factor_scores
```

### Documentation

- Use **Google-style docstrings**
- Include **type information** in docstrings
- Add **examples** for complex functions
- Update **README.md** for significant changes
- Update **research notebook** for new features

### Testing

- Write **unit tests** for all new functions
- Aim for **>90% test coverage**
- Include **edge cases** and error conditions
- Use **descriptive test names**
- Mock external dependencies (API calls, file I/O)

### Example Test

```python
def test_calculate_momentum_factor(self):
    """Test momentum factor calculation with various inputs"""
    # Setup test data
    test_data = self.create_sample_price_data(
        tickers=['AAPL', 'MSFT'], 
        start_date='2020-01-01',
        periods=500
    )
    
    calculator = FactorCalculator(test_data)
    momentum = calculator.calculate_momentum(lookback=252)
    
    # Assertions
    self.assertIsInstance(momentum, pd.Series)
    self.assertGreater(len(momentum.dropna()), 0)
    self.assertTrue(momentum.between(-5, 5).all())  # Reasonable range
```

## 🏗️ Architecture Guidelines

### Adding New Factors

1. **Extend FactorCalculator class**
2. **Follow naming convention**: `calculate_<factor_name>`
3. **Return pd.Series** with same index as input data
4. **Handle missing data** gracefully
5. **Add comprehensive tests**

```python
def calculate_custom_factor(self, window: int = 60) -> pd.Series:
    """Calculate custom factor"""
    # Validate inputs
    if window < 1:
        raise ValueError("Window must be positive")
    
    # Calculate factor
    factor_values = self.data.groupby('ticker')['Close'].rolling(
        window=window, min_periods=int(window * 0.8)
    ).apply(self._custom_calculation).reset_index(0, drop=True)
    
    return factor_values

def _custom_calculation(self, prices: pd.Series) -> float:
    """Helper method for custom calculation"""
    # Implementation details
    return result
```

### Adding New Data Sources

1. **Extend DataProvider class**
2. **Implement consistent interface**
3. **Handle rate limiting and errors**
4. **Add caching support**
5. **Document API requirements**

### Adding New Portfolio Construction Methods

1. **Extend PortfolioConstructor class**
2. **Return pd.Series with ticker weights**
3. **Respect position limits from config**
4. **Handle edge cases** (no data, single stock, etc.)

## 🐛 Bug Fixes

### Process

1. **Reproduce the bug** locally
2. **Write a failing test** that captures the bug
3. **Fix the bug** with minimal changes
4. **Ensure the test passes**
5. **Check for regressions**

### Guidelines

- **Minimal changes** to fix the specific issue
- **Preserve backwards compatibility** when possible
- **Add regression tests**
- **Update documentation** if behavior changes

## 📊 Performance Considerations

### Guidelines

- **Profile before optimizing** - measure actual bottlenecks
- **Use vectorized operations** with pandas/numpy
- **Implement caching** for expensive calculations
- **Consider memory usage** for large datasets
- **Add performance tests** for critical paths

### Benchmarking

```python
import time
import cProfile

def benchmark_function():
    """Benchmark critical function"""
    start_time = time.time()
    
    # Function to benchmark
    result = expensive_calculation()
    
    end_time = time.time()
    print(f"Execution time: {end_time - start_time:.2f} seconds")
    
    return result

# Profile with cProfile
cProfile.run('benchmark_function()')
```

## 📚 Documentation

### Requirements

- **Clear explanations** for all public APIs
- **Working examples** for complex features
- **Installation instructions**
- **Usage tutorials**
- **API reference**

### Jupyter Notebooks

- **Clear markdown explanations**
- **Working code examples**
- **Visualizations** where appropriate
- **Complete from start to finish**
- **Test notebooks** before committing

## 🚀 Release Process

### Version Numbering

We follow **Semantic Versioning** (semver):
- **MAJOR**: Breaking changes
- **MINOR**: New features (backwards compatible)
- **PATCH**: Bug fixes (backwards compatible)

### Checklist

- [ ] All tests pass
- [ ] Documentation updated
- [ ] CHANGELOG.md updated
- [ ] Version number bumped
- [ ] Git tag created
- [ ] PyPI package updated (if applicable)

## 💬 Community Guidelines

### Communication

- **Be respectful** and constructive
- **Stay on topic** in discussions
- **Help others** when possible
- **Ask questions** if something is unclear

### Code Review

- **Review thoroughly** but be constructive
- **Suggest improvements** rather than just pointing out problems
- **Explain reasoning** behind feedback
- **Approve when ready** - don't nitpick minor style issues

### Recognition

We recognize all contributors in:
- README.md contributors section
- CHANGELOG.md for significant contributions
- Release notes for major features

## ❓ Getting Help

### Resources

- **GitHub Issues**: Bug reports and feature requests
- **GitHub Discussions**: General questions and ideas
- **Documentation**: README.md and research notebook
- **Examples**: Check the examples/ directory

### Contact

For questions about contributing:
1. Check existing issues and discussions
2. Create a new issue with the "question" label
3. Reach out to maintainers directly for sensitive matters

---

Thank you for contributing to Factor Model Backtester! 🚀

Every contribution, no matter how small, helps make this project better for the entire community.
