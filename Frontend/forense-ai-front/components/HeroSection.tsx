import React from 'react';
import { AnalysisResult } from '../types';
import { ScoreGauge } from './ScoreGauge';
import { Badge } from './ui/Badge';
import { AlertTriangle, CheckCircle, HelpCircle, Bot, Activity } from 'lucide-react';

interface HeroSectionProps {
  data: AnalysisResult;
}

export const HeroSection: React.FC<HeroSectionProps> = ({ data }) => {
  const { automated_analysis, gemini_analysis } = data;
  const score = automated_analysis.final_score;

  // Semantic Logic
  let verdictColor = 'text-amber-500';
  let VerdictIcon = HelpCircle;
  let bgGradient = 'from-amber-500/10 to-transparent';

  if (score >= 0.55) {
    verdictColor = 'text-red-500';
    VerdictIcon = Bot;
    bgGradient = 'from-red-500/10 to-transparent';
  } else if (score <= 0.35) {
    verdictColor = 'text-emerald-500';
    VerdictIcon = CheckCircle;
    bgGradient = 'from-emerald-500/10 to-transparent';
  }

  return (
    <section className={`relative rounded-2xl bg-slate-800/50 border border-slate-700 overflow-hidden mb-8 shadow-xl animate-fade-in`}>
      {/* Background Gradient */}
      <div className={`absolute inset-0 bg-gradient-to-br ${bgGradient} opacity-30 pointer-events-none`} />

      <div className="relative p-6 md:p-10 grid grid-cols-1 lg:grid-cols-3 gap-8 items-center">
        
        {/* Left: Verdict & Gauge */}
        <div className="col-span-1 flex flex-col items-center lg:items-start text-center lg:text-left space-y-6">
          <div>
            <h2 className="text-sm font-bold text-slate-400 uppercase tracking-widest mb-2 flex items-center justify-center lg:justify-start gap-2">
              <Activity className="w-4 h-4" /> Veredicto Consolidado
            </h2>
            <div className={`text-4xl md:text-5xl font-extrabold ${verdictColor} flex items-center justify-center lg:justify-start gap-3`}>
              <VerdictIcon className="w-10 h-10 md:w-12 md:h-12" />
              {automated_analysis.interpretation}
            </div>
            <div className="mt-3 flex gap-2 justify-center lg:justify-start">
               <Badge variant={score >= 0.55 ? 'red' : score <= 0.35 ? 'green' : 'amber'}>
                 Confiança: {automated_analysis.confidence.toUpperCase()}
               </Badge>
               <Badge variant="blue">Gemini Pro Vision</Badge>
            </div>
          </div>
          
          <div className="lg:hidden">
             <ScoreGauge score={score} />
          </div>
        </div>

        {/* Middle: Gauge (Desktop) */}
        <div className="hidden lg:flex justify-center col-span-1">
          <ScoreGauge score={score} size={200} />
        </div>

        {/* Right: Gemini Explanation */}
        <div className="col-span-1 bg-slate-900/50 rounded-xl p-5 border border-slate-700/50 backdrop-blur-sm">
           <div className="flex items-center gap-2 mb-3 text-blue-400 font-semibold">
             <Bot className="w-5 h-5" />
             <span>Análise Gemini</span>
           </div>
           <p className="text-slate-300 text-sm leading-relaxed mb-4">
             {gemini_analysis.explanation}
           </p>
           <div className="space-y-2">
             <h4 className="text-xs font-bold text-slate-500 uppercase">Indicadores Chave</h4>
             <ul className="text-xs text-slate-400 space-y-1">
               {gemini_analysis.key_indicators.map((indicator, idx) => (
                 <li key={idx} className="flex items-start gap-2">
                   <span className="w-1 h-1 mt-1.5 rounded-full bg-blue-500 shrink-0" />
                   {indicator}
                 </li>
               ))}
             </ul>
           </div>
        </div>

      </div>
    </section>
  );
};