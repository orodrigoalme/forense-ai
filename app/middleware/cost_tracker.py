import os
import json
from datetime import datetime
from pathlib import Path
from typing import Tuple


class CostTracker:
    """
    Rastreia custos da API Gemini usando arquivo JSON local.
    Tracks Gemini API costs using a local JSON file.
    """
    
    def __init__(self):
        self.data_file = Path("cost_tracking.json")
        self.max_daily_cost = float(os.getenv("MAX_DAILY_GEMINI_COST", "5.0"))
        self.max_monthly_cost = float(os.getenv("MAX_MONTHLY_GEMINI_COST", "50.0"))
        self.cost_per_request = 0.002  # ~$0.002 por análise (ajustar conforme seu uso)
        
        self._load_data()
    
    def _load_data(self):
        """
        Carrega dados do arquivo.
        Loads data from file.
        """
        if self.data_file.exists():
            with open(self.data_file, 'r') as f:
                self.data = json.load(f)
        else:
            self.data = {"daily": {}, "monthly": {}}
    
    def _save_data(self):
        """
        Salva dados no arquivo.
        Saves data to file.
        """
        with open(self.data_file, 'w') as f:
            json.dump(self.data, f, indent=2)
    
    def can_make_request(self) -> Tuple[bool, str]:
        """
        Verifica se pode fazer requisição baseado no budget.
        Checks if a request can be made based on the budget.
        """
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
        """
        Registra custo de uma requisição.
        Records cost of a request.
        """
        today = datetime.now().strftime("%Y-%m-%d")
        month = datetime.now().strftime("%Y-%m")
        
        # Incrementar custo diário / Increment daily cost
        if today not in self.data["daily"]:
            self.data["daily"][today] = 0.0
        self.data["daily"][today] += self.cost_per_request
        
        # Incrementar custo mensal / Increment monthly cost
        if month not in self.data["monthly"]:
            self.data["monthly"][month] = 0.0
        self.data["monthly"][month] += self.cost_per_request
        
        # Limpar dados antigos / Clean old data (keep last 7 days and 3 months)
        self._cleanup_old_data()
        
        self._save_data()
    
    def _get_cost(self, period_type: str, key: str) -> float:
        """
        Obtém custo de um período.
        Gets cost for a period.
        """
        return self.data.get(period_type, {}).get(key, 0.0)
    
    def _cleanup_old_data(self):
        """
        Remove dados antigos para não encher o arquivo.
        Removes old data to keep the file small.
        """
        # Limpar dias antigos / Clean old days (keep last 7)
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
        
        # Limpar meses antigos / Clean old months (keep last 3)
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
        """
        Retorna uso atual.
        Returns current usage.
        """
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
