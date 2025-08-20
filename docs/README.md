# PRJ381 Documentation Strategy & Guide

This document serves as both a qui-**Status checking**: Verifies both Sphinx and FastAPI documentation
- **Flexible options**: Multiple build and serve configurations

## Documentation Types

### Interactive API Documentation guide and comprehensive documentation strategy for the PRJ381 Data Preprocessing API project.

## Documentation Architecture

The project uses a **hybrid documentation approach** that leverages the strengths of both FastAPI's automatic documentation and Sphinx's comprehensive documentation capabilities:

### 1. FastAPI Interactive Documentation (Primary API Reference)

**Purpose**: Live, interactive API documentation  
**Technology**: FastAPI + OpenAPI + Swagger UI/ReDoc  
**Access**: http://localhost:8000/docs (Swagger) or http://localhost:8000/redoc (ReDoc)

**Features**:
- Interactive endpoint testing
- Automatic request/response examples  
- Schema validation
- Real-time API exploration
- Always up-to-date with code

### 2. Sphinx Documentation (Comprehensive Project Documentation)

**Purpose**: Complete project documentation including guides, architecture, and detailed code reference  
**Technology**: Sphinx + autodoc + napoleon  
**Access**: Built HTML files (served locally or deployed)

**Features**:
- Rich formatting and cross-references
- Code documentation from docstrings
- User guides and tutorials
- Architecture documentation
- Testing guides
- PDF export capability

## Quick Start

### Build All Documentation

**Universal Script (All Platforms):**
```bash
python build_docs.py --serve --open
```

### Common Commands

**Build documentation:**
```bash
python build_docs.py
```

**Build and serve locally:**
```bash
python build_docs.py --serve
```

**Clean build and serve:**
```bash
python build_docs.py --clean --serve --open
```

**Skip Sphinx build (check FastAPI only):**
```bash
python build_docs.py --no-sphinx
```

**Custom port:**
```bash
python build_docs.py --serve --port 9000
```

### Script Features

- **Cross-platform**: Works on Windows, macOS, and Linux
- **Universal paths**: No hardcoded paths, works from any location
- **Dependency checking**: Automatically installs required packages
- **Error handling**: Comprehensive error reporting and recovery
- **Status checking**: Verifies both Sphinx and FastAPI documentation
- **Flexible options**: Multiple build and serve configurations

## Documentation Types

### Interactive API Documentation
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **Features**: Interactive testing, real-time schema validation

### Sphinx Documentation
- **Local Build**: `_build/html/index.html`
- **Features**: Comprehensive guides, architecture docs, detailed API reference

## Documentation Standards

### Docstring Format

Use **Google-style docstrings** that work well with both FastAPI and Sphinx:

```python
def example_function(param1: str, param2: int = 10) -> Dict[str, Any]:
    """
    Brief function description that appears in both FastAPI and Sphinx docs.
    
    Longer description with more details. This provides context and
    usage information that appears in Sphinx but not FastAPI.
    
    Args:
        param1 (str): Description of the first parameter
        param2 (int, optional): Description with default value. Defaults to 10.
        
    Returns:
        Dict[str, Any]: Description of return value with structure details
        
    Raises:
        HTTPException: When and why this exception is raised
        ValueError: When validation fails
        
    Example:
        Basic usage example::
        
            result = example_function("test", 20)
            print(result["key"])
            
    Note:
        Additional notes about usage, limitations, or important details.
    """
```

### FastAPI Route Documentation

FastAPI routes should include:

```python
@router.get("/endpoint", response_model=ResponseModel)
async def endpoint_function(
    param: str = Query(..., description="Parameter description for OpenAPI")
):
    """
    Brief endpoint description for FastAPI summary.
    
    Detailed description that explains what the endpoint does,
    when to use it, and any important considerations.
    
    Args:
        param (str): Detailed parameter description for Sphinx
        
    Returns:
        ResponseModel: Description of response structure
        
    Raises:
        HTTPException: 400 for invalid parameters
        HTTPException: 404 when resource not found
        
    Example:
        Request example::
        
            GET /endpoint?param=value
            
        Response::
        
            {"result": "success"}
    """
```

## For Developers

### Docstring Standards

Use Google-style docstrings compatible with both FastAPI and Sphinx:

```python
def example_endpoint(param: str) -> dict:
    """
    Brief description for FastAPI summary.
    
    Detailed description for Sphinx documentation with examples,
    usage notes, and comprehensive parameter documentation.
    
    Args:
        param (str): Parameter description
        
    Returns:
        dict: Response structure description
        
    Example:
        Usage example::
        
            GET /endpoint?param=value
    """
```

### File Structure

```
docs/
├── conf.py                 # Sphinx configuration
├── index.rst              # Main documentation index
├── api_reference.rst       # API documentation
├── user_guide.rst         # User guides
├── architecture.rst       # System architecture
├── testing.rst           # Testing documentation
├── README.md              # This comprehensive strategy & guide
└── _build/                # Generated documentation
    └── html/
        └── index.html     # Main documentation entry

# Universal build script in project root
build_docs.py              # Cross-platform documentation builder
```

### Configuration Details

#### Sphinx Configuration (docs/conf.py)

Key extensions and settings:

- `sphinx.ext.autodoc`: Auto-generate documentation from docstrings
- `sphinx.ext.napoleon`: Support for Google/NumPy style docstrings
- `sphinx_autodoc_typehints`: Include type hints in documentation
- `sphinx.ext.intersphinx`: Link to external documentation
- `sphinx.ext.extlinks`: Define shortcuts for common links

#### FastAPI Configuration (app/main.py)

Enhanced FastAPI app configuration:

- Detailed description with markdown support
- Comprehensive tag definitions
- Contact and license information
- Custom OpenAPI schema customization

## Cross-References & Integration

The documentation systems cross-reference each other:

### From Sphinx to FastAPI

Sphinx documentation includes links to live API documentation:

```rst
See the :swagger:`interactive API documentation </docs>` for testing endpoints.
```

### From FastAPI to Sphinx

FastAPI descriptions reference comprehensive documentation:

```python
description="""
## API Overview
For complete documentation including guides and examples, 
see the [Full Documentation](http://localhost:8080) when available.
"""
```

## Requirements

All documentation dependencies are included in `requirements.txt`:

- `sphinx>=8.2.3`
- `sphinx-rtd-theme>=3.0.2`
- `sphinx-autodoc-typehints>=2.6.0`

## Best Practices & Guidelines

### Docstring Guidelines

1. **First line**: Brief, action-oriented summary (shows in FastAPI)
2. **Description**: Detailed explanation (Sphinx only)
3. **Args**: Document all parameters with types
4. **Returns**: Describe return value structure
5. **Raises**: List possible exceptions
6. **Example**: Include usage examples
7. **Note/Warning**: Important considerations

### Consistency Rules

1. Use consistent terminology across both documentation systems
2. Ensure parameter descriptions match between Query/Path/Body and docstrings
3. Keep API summaries concise but descriptive
4. Include examples in docstrings for complex operations
5. Update both documentation types when making API changes

### Maintenance Workflow

1. Build documentation regularly during development
2. Review both Sphinx and FastAPI outputs for consistency
3. Test interactive examples in FastAPI docs
4. Validate cross-references and links
5. Keep dependencies in sync (requirements.txt)

## Deployment Considerations

### Production Documentation

For production deployments:

1. **FastAPI docs**: Always available at `/docs` and `/redoc` endpoints
2. **Sphinx docs**: Build and deploy to static hosting (GitHub Pages, etc.)
3. **Cross-links**: Update URLs to point to production documentation

### CI/CD Integration

Consider automating:

1. Documentation building in CI pipeline
2. Link checking and validation
3. Deployment to documentation hosting
4. API documentation testing

## Troubleshooting

### Common Issues

1. **Import errors**: Ensure all dependencies are installed
2. **Missing docstrings**: Check autodoc configuration
3. **Broken cross-references**: Verify intersphinx mappings
4. **FastAPI not reflecting changes**: Restart development server

### Debugging Steps

1. Check Sphinx build warnings for missing references
2. Validate FastAPI OpenAPI schema at `/openapi.json`
3. Test documentation links manually
4. Review console output for build errors

### Manual Build (If Script Fails)

```bash
# Manual Sphinx build
sphinx-build -b html docs/ docs/_build/html

# Start FastAPI server for interactive docs
uvicorn app.main:app --reload
```

---

## Complete Documentation Strategy Summary

This unified guide covers:

- **Quick start instructions** for immediate use  
- **Comprehensive strategy** for long-term maintenance  
- **Best practices** for consistent documentation  
- **Configuration details** for customization  
- **Troubleshooting guide** for common issues  
- **Deployment considerations** for production use

The hybrid FastAPI + Sphinx approach provides both immediate interactive testing capabilities and comprehensive project documentation, ensuring your API is well-documented for both users and developers.
