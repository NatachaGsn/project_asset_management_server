#!/bin/bash
cd /home/ubuntu/project_asset_management_server
source venv/bin/activate
python scripts/daily_report.py >> /home/ubuntu/project_asset_management_server/reports/cron.log 2>&1
