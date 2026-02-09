import React, { useState } from 'react';
import { AnalysisResult, MethodDetail, MethodMetrics } from '../types';
import { Badge } from './ui/Badge';
import { AlertTriangle, Check, Info, ZoomIn } from 'lucide-react';

interface DetailedAnalysisProps {
  data: AnalysisResult;
}

const MetricsDisplay = ({ metrics }: { metrics: MethodMetrics }) => {
  return (
    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
      {Object.entries(metrics).map(([key, value]) => {
        // Format Key
        const formattedKey = key.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
        
        // Format Value
        let formattedValue: React.ReactNode = String(value);
        if (typeof value === 'boolean') {
            formattedValue = value ? <span className="text-red-400 font-bold">TRUE</span> : <span className="text-emerald-400 font-bold">FALSE</span>;
        } else if (Array.isArray(value)) {
            formattedValue = `[${value.join(', ')}]`;
        } else if (typeof value === 'number') {
            formattedValue = <span className="font-mono text-blue-300">{value.toFixed(4)}</span>;
        } else if (value === null) {
            formattedValue = <span className="text-slate-600">N/A</span>;
        }

        return (
          <div key={key} className="flex justify-between items-center p-3 bg-slate-900/50 rounded border border-slate-700/50">
            <span className="text-xs text-slate-400 font-medium">{formattedKey}</span>
            <div className="text-sm font-medium text-slate-200 text-right">{formattedValue}</div>
          </div>
        );
      })}
    </div>
  );
};

const MethodTab = ({ detail }: { detail: MethodDetail }) => {
  return (
    <div className="animate-fade-in grid grid-cols-1 xl:grid-cols-2 gap-8">
      {/* Left: Image Visualization */}
      <div className="space-y-4">
        <div className="relative group rounded-lg overflow-hidden border border-slate-700 bg-slate-950 aspect-video flex items-center justify-center">
            {detail.image_base64 ? (
                <img 
                    src={detail.image_base64.startsWith('http') ? detail.image_base64 : `data:image/png;base64,${detail.image_base64}`} 
                    alt={`${detail.method} Analysis`} 
                    className="object-contain w-full h-full max-h-[400px]"
                />
            ) : (
                <div className="text-slate-600 flex flex-col items-center">
                    <Info className="w-8 h-8 mb-2" />
                    <span className="text-sm">Sem visualização disponível</span>
                </div>
            )}
            <div className="absolute top-2 right-2 opacity-0 group-hover:opacity-100 transition-opacity">
                <button className="bg-slate-900/80 p-2 rounded text-white hover:bg-slate-800 transition-colors">
                    <ZoomIn className="w-5 h-5" />
                </button>
            </div>
            <div className="absolute bottom-2 left-2">
                <Badge variant={detail.risk_score > 0.5 ? 'red' : 'green'}>
                    Risk Score: {detail.risk_score.toFixed(2)}
                </Badge>
            </div>
        </div>
        
        {/* Warnings */}
        <div className="bg-slate-900/30 border border-slate-800 rounded-lg p-4">
            <h4 className="text-sm font-semibold text-slate-300 mb-3 flex items-center gap-2">
                <AlertTriangle className="w-4 h-4 text-amber-500" />
                Observações do Algoritmo
            </h4>
            {detail.warnings.length > 0 ? (
                <ul className="space-y-2">
                    {detail.warnings.map((w, idx) => (
                        <li key={idx} className="text-sm text-amber-200/80 flex items-start gap-2 bg-amber-900/10 p-2 rounded border border-amber-900/20">
                            <span className="mt-1 w-1.5 h-1.5 rounded-full bg-amber-500 shrink-0"/>
                            {w}
                        </li>
                    ))}
                </ul>
            ) : (
                <div className="text-emerald-500 text-sm flex items-center gap-2">
                    <Check className="w-4 h-4" />
                    Nenhuma anomalia detectada.
                </div>
            )}
        </div>
      </div>

      {/* Right: Technical Metrics */}
      <div>
        <h3 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
            Métricas Técnicas
            <span className="text-xs font-normal text-slate-500 px-2 py-1 bg-slate-800 rounded">
                Raw Data
            </span>
        </h3>
        <MetricsDisplay metrics={detail.metrics} />
      </div>
    </div>
  );
};

export const DetailedAnalysis: React.FC<DetailedAnalysisProps> = ({ data }) => {
  const [activeTab, setActiveTab] = useState<'fft' | 'noise' | 'ela'>('fft');

  const tabs = [
    { id: 'fft', label: 'Espectro de Fourier (FFT)', desc: 'Analisa frequências e padrões repetitivos' },
    { id: 'noise', label: 'Análise de Ruído', desc: 'Verifica consistência do sensor' },
    { id: 'ela', label: 'Error Level Analysis', desc: 'Detecta manipulação de compressão' },
  ] as const;

  return (
    <section className="bg-slate-800/30 border border-slate-700/50 rounded-2xl overflow-hidden mt-8">
      <div className="border-b border-slate-700 bg-slate-900/50 px-4 md:px-6">
        <div className="flex gap-4 overflow-x-auto no-scrollbar">
          {tabs.map((tab) => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={`py-4 px-2 border-b-2 transition-colors whitespace-nowrap flex flex-col items-start ${
                activeTab === tab.id
                  ? 'border-blue-500 text-blue-400'
                  : 'border-transparent text-slate-400 hover:text-slate-200 hover:border-slate-700'
              }`}
            >
              <span className="text-sm font-bold">{tab.label}</span>
              <span className="text-[10px] opacity-70 hidden md:inline">{tab.desc}</span>
            </button>
          ))}
        </div>
      </div>

      <div className="p-6 min-h-[500px]">
        {data.details && <MethodTab detail={data.details[activeTab]} />}
      </div>
    </section>
  );
};