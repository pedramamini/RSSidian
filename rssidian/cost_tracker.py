"""
Cost tracking module for RSSidian.

This module provides functionality to track and calculate costs for API calls.
"""

import logging
import threading
import requests
import json
import os
from typing import Dict, Optional, Any, List
from decimal import Decimal
from functools import lru_cache

logger = logging.getLogger(__name__)

# Thread-local storage for cost tracking
_local = threading.local()

# Model pricing cache
_model_pricing = {}

# Anthropic model pricing (per million tokens)
ANTHROPIC_PRICING = {
    "claude-3-5-sonnet-20241022": {
        "input": 3.00,   # $3.00 per million input tokens
        "output": 15.00  # $15.00 per million output tokens
    },
    "claude-3-5-haiku-20241022": {
        "input": 1.00,   # $1.00 per million input tokens
        "output": 5.00   # $5.00 per million output tokens
    },
    "claude-3-opus-20240229": {
        "input": 15.00,  # $15.00 per million input tokens
        "output": 75.00  # $75.00 per million output tokens
    },
    "claude-3-sonnet-20240229": {
        "input": 3.00,   # $3.00 per million input tokens
        "output": 15.00  # $15.00 per million output tokens
    },
    "claude-3-haiku-20240307": {
        "input": 0.25,   # $0.25 per million input tokens
        "output": 1.25   # $1.25 per million output tokens
    }
}

@lru_cache(maxsize=32)
def get_model_pricing(model_id: str) -> Dict[str, Decimal]:
    """
    Get the pricing information for a model from OpenRouter.
    
    Args:
        model_id: The model ID to get pricing for
        
    Returns:
        A dictionary with 'prompt' and 'completion' pricing per token
    """
    # Default pricing if we can't get it from OpenRouter
    default_pricing = {
        'prompt': Decimal('0.000005'),  # $0.000005 per token
        'completion': Decimal('0.000015')  # $0.000015 per token
    }
    
    # Check if we already have the pricing in our cache
    if model_id in _model_pricing:
        return _model_pricing[model_id]
    
    # Try to get the pricing from OpenRouter
    try:
        response = requests.get('https://openrouter.ai/api/v1/models')
        if response.status_code == 200:
            models_data = response.json().get('data', [])
            for model_data in models_data:
                if model_data['id'] == model_id:
                    pricing = model_data.get('pricing', {})
                    _model_pricing[model_id] = {
                        'prompt': Decimal(str(pricing.get('prompt', default_pricing['prompt']))),
                        'completion': Decimal(str(pricing.get('completion', default_pricing['completion'])))
                    }
                    return _model_pricing[model_id]
    except Exception as e:
        logger.warning(f"Failed to get model pricing from OpenRouter: {str(e)}")
    
    # If we couldn't get the pricing, use the default
    _model_pricing[model_id] = default_pricing
    return default_pricing

def init_cost_tracker():
    """Initialize the cost tracker for the current thread."""
    _local.costs = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "prompt_cost": Decimal("0.0"),
        "completion_cost": Decimal("0.0"),
        "total_cost": Decimal("0.0"),
        "api_calls": 0,
        "models_used": {}
    }

def get_costs() -> Dict[str, Any]:
    """Get the current costs."""
    if not hasattr(_local, "costs"):
        init_cost_tracker()
    return _local.costs

def track_api_call(response_data: Dict[str, Any], model: str):
    """
    Track the cost of an API call.
    
    Args:
        response_data: The response data from the API call
        model: The model used for the API call
    """
    if not hasattr(_local, "costs"):
        init_cost_tracker()
    
    # Extract usage information from the response
    usage = response_data.get("usage", {})
    prompt_tokens = usage.get("prompt_tokens", 0)
    completion_tokens = usage.get("completion_tokens", 0)
    total_tokens = usage.get("total_tokens", 0)
    
    # If usage information is not available, estimate based on response content
    if not usage:
        # Rough estimation if usage is not provided
        choices = response_data.get("choices", [])
        if choices and len(choices) > 0:
            completion_content = choices[0].get("message", {}).get("content", "")
            # Rough estimate: 1 token ≈ 4 characters
            completion_tokens = len(completion_content) // 4
            # Assume prompt tokens based on completion (typically prompt is 1-2x completion)
            prompt_tokens = completion_tokens * 2
            total_tokens = prompt_tokens + completion_tokens
    
    # Get pricing for this model
    pricing = get_model_pricing(model)
    
    # Calculate costs
    prompt_cost = Decimal(prompt_tokens) * pricing['prompt']
    completion_cost = Decimal(completion_tokens) * pricing['completion']
    total_cost = prompt_cost + completion_cost
    
    # Update the cost tracker
    _local.costs["prompt_tokens"] += prompt_tokens
    _local.costs["completion_tokens"] += completion_tokens
    _local.costs["total_tokens"] += total_tokens
    _local.costs["prompt_cost"] += prompt_cost
    _local.costs["completion_cost"] += completion_cost
    _local.costs["total_cost"] += total_cost
    _local.costs["api_calls"] += 1
    
    # Track usage by model
    if model not in _local.costs["models_used"]:
        _local.costs["models_used"][model] = {
            "calls": 0,
            "tokens": 0,
            "cost": Decimal("0.0")
        }
    _local.costs["models_used"][model]["calls"] += 1
    _local.costs["models_used"][model]["tokens"] += total_tokens
    _local.costs["models_used"][model]["cost"] += total_cost
    
    logger.debug(f"API call to {model}: {prompt_tokens} prompt tokens, {completion_tokens} completion tokens, ${total_cost:.6f}")


def track_anthropic_api_call(response_data: Dict[str, Any], model: str):
    """
    Track the cost of an Anthropic API call.
    
    Args:
        response_data: The response data from the Anthropic API call
        model: The model used for the API call
    """
    if not hasattr(_local, "costs"):
        init_cost_tracker()
    
    # Extract usage information from the Anthropic response
    usage = response_data.get("usage", {})
    input_tokens = usage.get("input_tokens", 0)
    output_tokens = usage.get("output_tokens", 0)
    total_tokens = input_tokens + output_tokens
    
    # Get pricing for this model
    pricing = ANTHROPIC_PRICING.get(model)
    if not pricing:
        logger.warning(f"No pricing information for Anthropic model {model}, using default")
        pricing = {"input": 3.00, "output": 15.00}  # Default to Sonnet pricing
    
    # Calculate costs (pricing is per million tokens)
    input_cost = Decimal(input_tokens) * Decimal(str(pricing["input"])) / Decimal("1000000")
    output_cost = Decimal(output_tokens) * Decimal(str(pricing["output"])) / Decimal("1000000")
    total_cost = input_cost + output_cost
    
    # Update the cost tracker
    _local.costs["prompt_tokens"] += input_tokens
    _local.costs["completion_tokens"] += output_tokens
    _local.costs["total_tokens"] += total_tokens
    _local.costs["prompt_cost"] += input_cost
    _local.costs["completion_cost"] += output_cost
    _local.costs["total_cost"] += total_cost
    _local.costs["api_calls"] += 1
    
    # Track usage by model
    if model not in _local.costs["models_used"]:
        _local.costs["models_used"][model] = {
            "calls": 0,
            "tokens": 0,
            "cost": Decimal("0.0")
        }
    _local.costs["models_used"][model]["calls"] += 1
    _local.costs["models_used"][model]["tokens"] += total_tokens
    _local.costs["models_used"][model]["cost"] += total_cost
    
    logger.debug(f"Anthropic API call to {model}: {input_tokens} input tokens, {output_tokens} output tokens, ${total_cost:.6f}")

def format_cost_summary() -> str:
    """
    Format the cost summary as a string.
    Only returns a summary if there are non-zero costs.
    
    Returns:
        A formatted string with the cost summary if costs exist, empty string otherwise
    """
    if not hasattr(_local, "costs"):
        return ""
    
    costs = _local.costs
    
    # Return empty string if no actual costs
    if costs['total_cost'] == Decimal("0.0"):
        return ""
    
    summary = [
        "AI Cost Summary:",
        f"  API_Calls:: {costs['api_calls']}",
        f"  Prompt_Tokens:: {costs['prompt_tokens']}",
        f"  Completion_Tokens:: {costs['completion_tokens']}",
        f"  Total_Tokens:: {costs['total_tokens']}",
        f"  Total_Cost:: {costs['total_cost']:.6f}"
    ]
    
    # Add model-specific information if available
    if costs['models_used']:
        summary.append("\nCost by Model:")
        for model, model_costs in costs['models_used'].items():
            summary.append(f"  {model}:")
            summary.append(f"    Calls: {model_costs['calls']}")
            summary.append(f"    Tokens: {model_costs['tokens']}")
            summary.append(f"    Cost: {model_costs['cost']:.6f}")
    
    return "\n".join(summary)
