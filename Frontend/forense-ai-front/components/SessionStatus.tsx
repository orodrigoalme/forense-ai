import React, { useState } from 'react';
import { SessionStats } from '../types';
import { Shield, Battery, BatteryMedium, BatteryLow, RefreshCw, Key, LogOut, WifiOff, Star } from 'lucide-react';
import { authService } from '../services/authService';

interface SessionStatusProps {
  stats: SessionStats | null;
  onRefresh: () => void;
}

export const SessionStatus: React.FC<SessionStatusProps> = ({ stats, onRefresh }) => {
  const [isOpen, setIsOpen] = useState(false);
  const [isResetting, setIsResetting] = useState(false);

  if (!stats) return null;

  const handleResetSession = async () => {
    setIsResetting(true);
    await authService.deleteSession();
    try {
        await authService.initSession();
    } catch (e) {
        console.error("Error re-initializing session:", e);
    }
    onRefresh();
    setIsResetting(false);
    setIsOpen(false);
  };

  const remaining = stats.requests_remaining;
  const isOffline = stats.limit_type === 'offline_demo';
  const isCustomKey = stats.limit_type === 'custom_key';
  
  let BatteryIcon = Battery;
  let colorClass = 'text-emerald-400';
  let badgeColor = 'bg-emerald-500';

  if (isOffline) {
    colorClass = 'text-slate-400';
    badgeColor = 'bg-slate-500';
  } else if (isCustomKey) {
    colorClass = 'text-blue-400';
    badgeColor = 'bg-blue-500';
  } else if (remaining < 10) {
    BatteryIcon = BatteryLow;
    colorClass = 'text-red-400';
    badgeColor = 'bg-red-500';
  } else if (remaining < 25) {
    BatteryIcon = BatteryMedium;
    colorClass = 'text-amber-400';
    badgeColor = 'bg-amber-500';
  }

  const getLabel = () => {
      if (isOffline) return 'Modo Offline';
      if (isCustomKey) return 'Chave Própria';
      return 'Sessão Anônima';
  };

  return (
    <div className="relative z-50">
      <button 
        onClick={() => setIsOpen(!isOpen)}
        className={`flex items-center gap-2 px-3 py-1.5 border rounded-full text-xs font-medium transition-all duration-300
            ${isCustomKey 
                ? 'bg-blue-500/10 border-blue-500/30 hover:bg-blue-500/20' 
                : 'bg-slate-800/80 border-slate-700 hover:bg-slate-800'
            }
        `}
      >
        <div className={`w-2 h-2 rounded-full ${!isOffline && !isCustomKey && remaining > 0 ? `${badgeColor} animate-pulse` : badgeColor}`} />
        
        <span className={`hidden sm:inline ${isCustomKey ? 'text-blue-300 font-semibold' : 'text-slate-300'}`}>
          {getLabel()}
        </span>

        {/* Show counts ONLY if NOT offline and NOT custom key */}
        {!isOffline && !isCustomKey && (
            <span className={`${colorClass} font-mono ml-1`}>
                {remaining} left
            </span>
        )}
        
        {isOffline && <WifiOff className="w-3 h-3 text-slate-500 ml-1" />}
        {isCustomKey && <Key className="w-3 h-3 text-blue-400 ml-1" />}
      </button>

      {isOpen && (
        <>
          <div className="fixed inset-0 cursor-default" onClick={() => setIsOpen(false)} />
          <div className="absolute right-0 mt-2 w-72 bg-slate-900 border border-slate-700 rounded-xl shadow-2xl overflow-hidden animate-fade-in origin-top-right">
            <div className="p-4 border-b border-slate-800">
              <h4 className="text-sm font-bold text-white flex items-center gap-2">
                <Shield className="w-4 h-4 text-blue-500" />
                Detalhes da Sessão Segura
              </h4>
            </div>
            
            <div className="p-4 space-y-4">
              {isCustomKey ? (
                  <div className="bg-blue-500/10 border border-blue-500/20 rounded-lg p-4 text-center animate-fade-in">
                    <div className="flex justify-center mb-3">
                        <div className="p-2.5 bg-blue-500/20 rounded-full ring-2 ring-blue-500/20">
                            <Star className="w-5 h-5 text-blue-400 fill-blue-500/20" />
                        </div>
                    </div>
                    <h5 className="text-sm font-bold text-blue-300 mb-2">Modo Avançado Ativo</h5>
                    <p className="text-xs text-blue-200/70 leading-relaxed">
                        Você está utilizando sua própria chave Gemini. Os limites de requisição da sessão padrão não se aplicam e os contadores estão ocultos.
                    </p>
                  </div>
              ) : (
                  <div className="grid grid-cols-2 gap-2">
                     <div className="bg-slate-800/50 p-2 rounded border border-slate-700">
                        <span className="text-[10px] text-slate-500 uppercase block">Requisições</span>
                        {isOffline ? (
                            <span className="text-sm text-slate-400 italic">Ilimitado (Local)</span>
                        ) : (
                            <>
                                <span className={`text-xl font-mono ${colorClass}`}>{stats.requests_remaining}</span>
                                <span className="text-[10px] text-slate-500"> / {stats.requests_used + stats.requests_remaining}</span>
                            </>
                        )}
                     </div>
                     <div className="bg-slate-800/50 p-2 rounded border border-slate-700">
                        <span className="text-[10px] text-slate-500 uppercase block">Tipo Limite</span>
                        <span className={`text-xs font-semibold capitalize ${isOffline ? 'text-slate-400' : 'text-blue-300'}`}>
                            {stats.limit_type.replace('_', ' ')}
                        </span>
                     </div>
                  </div>
              )}
              
              {isOffline && (
                  <div className="flex items-center gap-2 text-xs text-slate-400 bg-slate-800 p-2 rounded border border-slate-700">
                      <WifiOff className="w-3 h-3" />
                      <span>Modo Demonstração (Sem conexão)</span>
                  </div>
              )}

              <div className="text-[10px] text-slate-500 font-mono break-all pt-2 border-t border-slate-800/50 mt-2">
                ID: {stats.session_id || 'offline-session'}
              </div>
            </div>

            <div className="bg-slate-800/50 p-3 border-t border-slate-800 flex justify-between">
              <button 
                onClick={() => { onRefresh(); setIsOpen(false); }}
                className="flex items-center gap-1.5 text-xs text-slate-400 hover:text-white transition-colors"
              >
                <RefreshCw className="w-3 h-3" />
                Atualizar
              </button>
              
              <button 
                onClick={handleResetSession}
                disabled={isResetting}
                className="flex items-center gap-1.5 text-xs text-red-400 hover:text-red-300 transition-colors disabled:opacity-50"
              >
                {isResetting ? 'Recriando...' : (
                    <>
                        <LogOut className="w-3 h-3" />
                        Nova Sessão
                    </>
                )}
              </button>
            </div>
          </div>
        </>
      )}
    </div>
  );
};