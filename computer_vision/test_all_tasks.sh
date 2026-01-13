#!/bin/bash
# Quick test script for all tasks

echo "=========================================="
echo "Testing All Tasks - Step by Step"
echo "=========================================="

# Navigate to project directory
cd /home/ros_master/computer_vision

echo ""
echo "Step 1: Testing Task 1 (Object Detection)"
echo "----------------------------------------"
python3 run_task.py --task task1 --mode demo

echo ""
echo "Step 2: Testing Task 2 (Quality Inspection)"
echo "----------------------------------------"
python3 run_task.py --task task2 --mode demo

echo ""
echo "Step 3: Testing Task 3 (VLM Design)"
echo "----------------------------------------"
python3 run_task.py --task task3

echo ""
echo "=========================================="
echo "All tasks tested!"
echo "=========================================="
