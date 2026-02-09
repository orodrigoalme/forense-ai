# app/middleware/quota.py

import os
from datetime import datetime, timedelta
from typing import Dict

class QuotaManager:
    """Sistema de quotas por usuário"""
    
    def __init__(self):
        # Quotas diárias por API Key
        self.quotas = {}
        self.free_tier_limit = int(os.getenv("FREE_TIER_DAILY_LIMIT", "10"))
        self.premium_tier_limit = int(os.getenv("PREMIUM_TIER_DAILY_LIMIT", "100"))
        
        # Whitelist de chaves premium (definir no .env)
        premium_keys = os.getenv("PREMIUM_API_KEYS", "")
        self.premium_keys = set(k.strip() for k in premium_keys.split(',') if k.strip())
    
    def check_quota(self, api_key: str) -> tuple[bool, str]:
        """Verifica se usuário ainda tem quota"""
        today = datetime.now().date()
        
        if api_key not in self.quotas:
            self.quotas[api_key] = {"date": today, "count": 0}
        
        quota_data = self.quotas[api_key]
        
        # Reset se mudou o dia
        if quota_data["date"] != today:
            quota_data["date"] = today
            quota_data["count"] = 0
        
        # Verificar limite
        is_premium = api_key in self.premium_keys
        limit = self.premium_tier_limit if is_premium else self.free_tier_limit
        
        if quota_data["count"] >= limit:
            tier = "premium" if is_premium else "free"
            return False, f"Quota diária excedida ({limit} requisições no tier {tier})"
        
        return True, "OK"
    
    def consume_quota(self, api_key: str):
        """Consome 1 quota"""
        self.quotas[api_key]["count"] += 1

quota_manager = QuotaManager()