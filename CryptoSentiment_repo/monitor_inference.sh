#!/bin/bash
# monitor_inference.sh - Monitor MPS GPU usage during inference

echo "🔍 Starting MPS GPU monitoring for checkpoint inference..."
echo "Press Ctrl+C to stop monitoring"

# Function to run monitoring in background
monitor_gpu() {
    while true; do
        echo "=== $(date) ==="
        
        # GPU utilization and power
        sudo powermetrics --samplers gpu_power -n 1 -i 1000 2>/dev/null | \
            grep -E "(GPU HW active frequency|GPU idle frequency|GPU|Combined Power)" | \
            head -10
        
        # Process info for inference
        ps aux | grep -E "(checkpointed_inference|python.*inference)" | grep -v grep
        
        # Memory usage
        vm_stat | grep -E "(Pages free|Pages active|Pages inactive)" | \
            awk '{print $1 $2 $3}' | head -3
        
        echo "---"
        sleep 3
    done
}

# Start monitoring
monitor_gpu 