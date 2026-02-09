import React from 'react';

type BadgeProps = {
  children: React.ReactNode;
  variant?: 'red' | 'green' | 'amber' | 'blue' | 'gray';
  className?: string;
};

export const Badge: React.FC<BadgeProps> = ({ children, variant = 'gray', className = '' }) => {
  const styles = {
    red: 'bg-red-500/10 text-red-400 border-red-500/20',
    green: 'bg-emerald-500/10 text-emerald-400 border-emerald-500/20',
    amber: 'bg-amber-500/10 text-amber-400 border-amber-500/20',
    blue: 'bg-blue-500/10 text-blue-400 border-blue-500/20',
    gray: 'bg-slate-700/50 text-slate-300 border-slate-600',
  };

  return (
    <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium border ${styles[variant]} ${className}`}>
      {children}
    </span>
  );
};