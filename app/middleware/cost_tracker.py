import os
import json
from datetime import datetime
from pathlib import Path
from typing import Tuple


class CostTracker:
    """Rastreia custos da API Gemini usando arquivo JSON local"""
    
    def __init__(self):
        self.data_file = Path("cost_tracking.json")
        self.max_daily_cost = float(os.getenv("MAX_DAILY_GEMINI_COST", "5.0"))
        self.max_monthly_cost = float(os.getenv("MAX_MONTHLY_GEMINI_COST", "50.0"))
        self.cost_per_request = 0.002  # ~$0.002 por análise (ajustar conforme seu uso)
        
        self._load_data()
    
    def _load_data(self):
        """Carrega dados do arquivo"""
        if self.data_file.exists():
            with open(self.data_file, 'r') as f:
                self.data = json.load(f)
        else:
            self.data = {"daily": {}, "monthly": {}}
    
    def _save_data(self):
        """Salva dados no arquivo"""
        with open(self.data_file, 'w') as f:
            json.dump(self.data, f, indent=2)
    
    def can_make_request(self) -> Tuple[bool, str]:
        """Verifica se pode fazer requisição baseado no budget"""
        today = datetime.now().strftime("%Y-%m-%d")
        month = datetime.now().strftime("%Y-%m")
        
        today_cost = self._get_cost("daily", today)
        month_cost = self._get_cost("monthly", month)
        
        if today_cost >= self.max_daily_cost:
            return False, f"Limite diário atingido (${today_cost:.2f}/${self.max_daily_cost:.2f})"
        
        if month_cost >= self.max_monthly_cost:
            return False, f"Limite mensal atingido (${month_cost:.2f}/${self.max_monthly_cost:.2f})"
        
        return True, "OK"
    
    def track_request(self):
        """Registra custo de uma requisição"""
        today = datetime.now().strftime("%Y-%m-%d")
        month = datetime.now().strftime("%Y-%m")
        
        # Incrementar custo diário
        if today not in self.data["daily"]:
            self.data["daily"][today] = 0.0
        self.data["daily"][today] += self.cost_per_request
        
        # Incrementar custo mensal
        if month not in self.data["monthly"]:
            self.data["monthly"][month] = 0.0
        self.data["monthly"][month] += self.cost_per_request
        
        # Limpar dados antigos (manter apenas últimos 7 dias e 3 meses)
        self._cleanup_old_data()
        
        self._save_data()
    
    def _get_cost(self, period_type: str, key: str) -> float:
        """Obtém custo de um período"""
        return self.data.get(period_type, {}).get(key, 0.0)
    
    def _cleanup_old_data(self):
        """Remove dados antigos para não encher o arquivo"""
        # Limpar dias antigos (manter últimos 7 dias)
        today = datetime.now()
        days_to_keep = 7
        
        daily_keys = list(self.data["daily"].keys())
        for day_str in daily_keys:
            try:
                day_date = datetime.strptime(day_str, "%Y-%m-%d")
                if (today - day_date).days > days_to_keep:
                    del self.data["daily"][day_str]
            except:
                pass
        
        # Limpar meses antigos (manter últimos 3 meses)
        months_to_keep = 3
        monthly_keys = list(self.data["monthly"].keys())
        for month_str in monthly_keys:
            try:
                month_date = datetime.strptime(month_str, "%Y-%m")
                months_diff = (today.year - month_date.year) * 12 + (today.month - month_date.month)
                if months_diff > months_to_keep:
                    del self.data["monthly"][month_str]
            except:
                pass
    
    def get_current_usage(self) -> dict:
        """Retorna uso atual (para dashboard/logs)"""
        today = datetime.now().strftime("%Y-%m-%d")
        month = datetime.now().strftime("%Y-%m")
        
        return {
            "today": {
                "used": self._get_cost("daily", today),
                "limit": self.max_daily_cost,
                "remaining": max(0, self.max_daily_cost - self._get_cost("daily", today))
            },
            "month": {
                "used": self._get_cost("monthly", month),
                "limit": self.max_monthly_cost,
                "remaining": max(0, self.max_monthly_cost - self._get_cost("monthly", month))
            }
        }
