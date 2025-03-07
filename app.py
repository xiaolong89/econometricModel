"""
Marketing Mix Model API

This module implements a Flask API for the Marketing Mix Model (MMM),
providing endpoints for file upload, model training, results retrieval,
and budget optimization.
"""

import os
import json
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
import pandas as pd
from model_wrapper import (
    save_uploaded_file,
    train_mmm_model,
    get_model_results,
    get_model_visualizations,
    get_model_diagnostics,
    optimize_budget
)

# Initialize Flask app
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50 MB max upload size
app.config['UPLOAD_FOLDER'] = './data'

# Enable CORS
CORS(app)

# Simple API key authentication (replace with proper auth in production)
API_KEYS = ['dev_key_123', 'test_key_456']

# Create upload folder if it doesn't exist
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])


def validate_api_key():
    """Validate API key from request header"""
    api_key = request.headers.get('X-API-Key')
    if not api_key or api_key not in API_KEYS:
        return False
    return True


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'version': '1.0.0'
    })


@app.route('/api/upload', methods=['POST'])
def upload_file():
    """Upload a CSV file for analysis"""
    # Check API key
    if not validate_api_key():
        return jsonify({
            'success': False,
            'error': {
                'code': 'UNAUTHORIZED',
                'message': 'Invalid or missing API key'
            }
        }), 401

    # Check if file was included in request
    if 'file' not in request.files:
        return jsonify({
            'success': False,
            'error': {
                'code': 'NO_FILE',
                'message': 'No file provided in request'
            }
        }), 400

    file = request.files['file']

    # Check if filename is empty
    if file.filename == '':
        return jsonify({
            'success': False,
            'error': {
                'code': 'EMPTY_FILENAME',
                'message': 'File has no name'
            }
        }), 400

    if file:
        try:
            # Secure the filename
            filename = secure_filename(file.filename)

            # Save the file and get metadata
            file_metadata = save_uploaded_file(file, filename)

            # Return success response with file metadata
            return jsonify({
                'success': True,
                'message': 'File uploaded successfully',
                'data': file_metadata
            })

        except Exception as e:
            return jsonify({
                'success': False,
                'error': {
                    'code': 'UPLOAD_ERROR',
                    'message': str(e)
                }
            }), 400

    return jsonify({
        'success': False,
        'error': {
            'code': 'UNKNOWN_ERROR',
            'message': 'An unknown error occurred'
        }
    }), 500


@app.route('/api/model/train', methods=['POST'])
def train_model_endpoint():
    """Train a model with the specified configuration"""
    # Check API key
    if not validate_api_key():
        return jsonify({
            'success': False,
            'error': {
                'code': 'UNAUTHORIZED',
                'message': 'Invalid or missing API key'
            }
        }), 401

    # Check request body
    if not request.json:
        return jsonify({
            'success': False,
            'error': {
                'code': 'INVALID_REQUEST',
                'message': 'Request body must be JSON'
            }
        }), 400

    # Extract required parameters
    file_id = request.json.get('file_id')
    config = request.json.get('config', {})

    # Validate parameters
    if not file_id:
        return jsonify({
            'success': False,
            'error': {
                'code': 'MISSING_FILE_ID',
                'message': 'file_id is required'
            }
        }), 400

    if not config:
        return jsonify({
            'success': False,
            'error': {
                'code': 'MISSING_CONFIG',
                'message': 'config is required'
            }
        }), 400

    # Validate essential config parameters
    required_params = ['target_variable', 'date_variable', 'media_variables']
    missing_params = [param for param in required_params if param not in config]

    if missing_params:
        return jsonify({
            'success': False,
            'error': {
                'code': 'MISSING_CONFIG_PARAMS',
                'message': f'Missing required configuration parameters: {", ".join(missing_params)}'
            }
        }), 400

    try:
        # Train the model
        model_info = train_mmm_model(file_id, config)

        # Return success response with model info
        return jsonify({
            'success': True,
            'message': 'Model trained successfully',
            'data': model_info
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': {
                'code': 'TRAINING_ERROR',
                'message': str(e)
            }
        }), 400


@app.route('/api/model/results/<model_id>', methods=['GET'])
def get_results_endpoint(model_id):
    """Get results for a trained model"""
    # Check API key
    if not validate_api_key():
        return jsonify({
            'success': False,
            'error': {
                'code': 'UNAUTHORIZED',
                'message': 'Invalid or missing API key'
            }
        }), 401

    try:
        # Get model results
        results = get_model_results(model_id)

        # Return success response with results
        return jsonify({
            'success': True,
            'message': 'Model results retrieved successfully',
            'data': results
        })

    except ValueError as e:
        return jsonify({
            'success': False,
            'error': {
                'code': 'MODEL_NOT_FOUND',
                'message': str(e)
            }
        }), 404

    except Exception as e:
        return jsonify({
            'success': False,
            'error': {
                'code': 'RESULTS_ERROR',
                'message': str(e)
            }
        }), 400


@app.route('/api/model/visualizations/<model_id>', methods=['GET'])
def get_visualizations_endpoint(model_id):
    """Get visualization data for a trained model"""
    # Check API key
    if not validate_api_key():
        return jsonify({
            'success': False,
            'error': {
                'code': 'UNAUTHORIZED',
                'message': 'Invalid or missing API key'
            }
        }), 401

    # Get visualization type filter if provided
    viz_type = request.args.get('type')

    try:
        # Get visualization data
        visualizations = get_model_visualizations(model_id, viz_type)

        # Return success response with visualization data
        return jsonify({
            'success': True,
            'message': 'Visualizations retrieved successfully',
            'data': visualizations
        })

    except ValueError as e:
        return jsonify({
            'success': False,
            'error': {
                'code': 'MODEL_NOT_FOUND',
                'message': str(e)
            }
        }), 404

    except Exception as e:
        return jsonify({
            'success': False,
            'error': {
                'code': 'VISUALIZATION_ERROR',
                'message': str(e)
            }
        }), 400


@app.route('/api/model/diagnostics/<model_id>', methods=['GET'])
def get_diagnostics_endpoint(model_id):
    """Get diagnostic results for a trained model"""
    # Check API key
    if not validate_api_key():
        return jsonify({
            'success': False,
            'error': {
                'code': 'UNAUTHORIZED',
                'message': 'Invalid or missing API key'
            }
        }), 401

    try:
        # Get diagnostic results
        diagnostics = get_model_diagnostics(model_id)

        # Return success response with diagnostic data
        return jsonify({
            'success': True,
            'message': 'Diagnostic results retrieved successfully',
            'data': diagnostics
        })

    except ValueError as e:
        return jsonify({
            'success': False,
            'error': {
                'code': 'MODEL_NOT_FOUND',
                'message': str(e)
            }
        }), 404

    except Exception as e:
        return jsonify({
            'success': False,
            'error': {
                'code': 'DIAGNOSTICS_ERROR',
                'message': str(e)
            }
        }), 400


@app.route('/api/model/optimize/<model_id>', methods=['POST'])
def optimize_budget_endpoint(model_id):
    """Optimize budget allocation based on model results"""
    # Check API key
    if not validate_api_key():
        return jsonify({
            'success': False,
            'error': {
                'code': 'UNAUTHORIZED',
                'message': 'Invalid or missing API key'
            }
        }), 401

    # Check request body
    if not request.json:
        return jsonify({
            'success': False,
            'error': {
                'code': 'INVALID_REQUEST',
                'message': 'Request body must be JSON'
            }
        }), 400

    try:
        # Run budget optimization
        optimization_results = optimize_budget(model_id, request.json)

        # Return success response with optimization results
        return jsonify({
            'success': True,
            'message': 'Budget optimization completed successfully',
            'data': optimization_results
        })

    except ValueError as e:
        return jsonify({
            'success': False,
            'error': {
                'code': 'MODEL_NOT_FOUND',
                'message': str(e)
            }
        }), 404

    except Exception as e:
        return jsonify({
            'success': False,
            'error': {
                'code': 'OPTIMIZATION_ERROR',
                'message': str(e)
            }
        }), 400


if __name__ == '__main__':
    # Run the Flask app (development server)
    app.run(host='0.0.0.0', port=5000, debug=True)