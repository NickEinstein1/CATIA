"""
CATIA Command Line Interface

Entry point for the CATIA package when installed via pip.
Usage: catia [options]
"""

import argparse
import sys
import logging

from catia import __version__
from catia.config import LOGGING_CONFIG, DEFAULT_PERILS


def setup_logging(verbose: bool = False):
    """Configure logging based on verbosity."""
    level = logging.DEBUG if verbose else LOGGING_CONFIG["level"]
    logging.basicConfig(
        level=level,
        format=LOGGING_CONFIG["format"]
    )


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="catia",
        description="CATIA: Catastrophe AI System for Climate Risk Modeling",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  catia --region US_Gulf_Coast --perils hurricane flood
  catia --api --port 8000
  catia --version
        """
    )
    
    parser.add_argument(
        "--version", "-V",
        action="version",
        version=f"CATIA v{__version__}"
    )
    
    parser.add_argument(
        "--region", "-r",
        default="US_Gulf_Coast",
        help="Geographic region for analysis (default: US_Gulf_Coast)"
    )
    
    parser.add_argument(
        "--perils", "-p",
        nargs="+",
        default=DEFAULT_PERILS,
        choices=["hurricane", "flood", "wildfire", "earthquake"],
        help="Perils to analyze (default: all)"
    )
    
    parser.add_argument(
        "--mock-data",
        action="store_true",
        default=True,
        help="Use mock data instead of real APIs (default: True)"
    )
    
    parser.add_argument(
        "--no-mock-data",
        action="store_true",
        help="Use real API data (requires API keys)"
    )
    
    parser.add_argument(
        "--api",
        action="store_true",
        help="Start the FastAPI server instead of running analysis"
    )
    
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="API server host (default: 0.0.0.0)"
    )
    
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="API server port (default: 8000)"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose/debug logging"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)
    
    # Handle API mode
    if args.api:
        try:
            import uvicorn
            from catia.api.app import app
            logger.info(f"Starting CATIA API server on {args.host}:{args.port}")
            uvicorn.run(app, host=args.host, port=args.port)
        except ImportError:
            logger.error("uvicorn not installed. Run: pip install uvicorn")
            sys.exit(1)
        return
    
    # Import and run main analysis (lazy import to speed up --help)
    # Import from root main.py - we import the function directly
    try:
        # Add parent directory to path for main.py import
        import os
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from main import run_catia_analysis
        
        use_mock = not args.no_mock_data
        
        logger.info(f"Running CATIA analysis...")
        logger.info(f"  Region: {args.region}")
        logger.info(f"  Perils: {args.perils}")
        logger.info(f"  Mock data: {use_mock}")
        
        results = run_catia_analysis(
            region=args.region,
            use_mock_data=use_mock,
            perils=args.perils
        )
        
        # Print summary
        print(f"\n{'='*60}")
        print("CATIA Analysis Complete")
        print(f"{'='*60}")
        print(f"Mean Annual Loss: ${results['risk_metrics']['descriptive_stats']['mean']:,.0f}")
        print(f"VaR (95%): ${results['risk_metrics']['risk_metrics']['var']:,.0f}")
        print(f"TVaR (95%): ${results['risk_metrics']['risk_metrics']['tvar']:,.0f}")
        print(f"{'='*60}")
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}", exc_info=args.verbose)
        sys.exit(1)


if __name__ == "__main__":
    main()

