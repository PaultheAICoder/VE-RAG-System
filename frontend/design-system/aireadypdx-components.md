# AI Ready PDX — React Component Library

## 1. Logo Component

```jsx
export const BridgeLogo = ({ size = 120, className = '' }) => (
  <svg 
    width={size} 
    height={size * 0.55} 
    viewBox="0 0 120 66" 
    fill="none"
    className={className}
  >
    <path d="M8 32 Q60 2 112 32" stroke="currentColor" strokeWidth="2" fill="none" strokeLinecap="round"/>
    <path d="M20 58 L20 24 L26 16 L32 24 L32 58" stroke="currentColor" strokeWidth="1.5" fill="none"/>
    <line x1="26" y1="16" x2="26" y2="58" stroke="currentColor" strokeWidth="1.5"/>
    <line x1="20" y1="28" x2="32" y2="28" stroke="currentColor" strokeWidth="1.5"/>
    <line x1="20" y1="36" x2="32" y2="36" stroke="currentColor" strokeWidth="1.5"/>
    <line x1="20" y1="44" x2="32" y2="44" stroke="currentColor" strokeWidth="1.5"/>
    <line x1="20" y1="52" x2="32" y2="52" stroke="currentColor" strokeWidth="1.5"/>
    <rect x="22" y="10" width="8" height="6" stroke="currentColor" strokeWidth="1.5" fill="none"/>
    <line x1="26" y1="6" x2="26" y2="10" stroke="currentColor" strokeWidth="1"/>
    <path d="M88 58 L88 24 L94 16 L100 24 L100 58" stroke="currentColor" strokeWidth="1.5" fill="none"/>
    <line x1="94" y1="16" x2="94" y2="58" stroke="currentColor" strokeWidth="1.5"/>
    <line x1="88" y1="28" x2="100" y2="28" stroke="currentColor" strokeWidth="1.5"/>
    <line x1="88" y1="36" x2="100" y2="36" stroke="currentColor" strokeWidth="1.5"/>
    <line x1="88" y1="44" x2="100" y2="44" stroke="currentColor" strokeWidth="1.5"/>
    <line x1="88" y1="52" x2="100" y2="52" stroke="currentColor" strokeWidth="1.5"/>
    <rect x="90" y="10" width="8" height="6" stroke="currentColor" strokeWidth="1.5" fill="none"/>
    <line x1="94" y1="6" x2="94" y2="10" stroke="currentColor" strokeWidth="1"/>
    <line x1="40" y1="12" x2="40" y2="46" stroke="currentColor" strokeWidth="1.2"/>
    <line x1="50" y1="7" x2="50" y2="46" stroke="currentColor" strokeWidth="1.2"/>
    <line x1="60" y1="4" x2="60" y2="46" stroke="currentColor" strokeWidth="1.2"/>
    <line x1="70" y1="7" x2="70" y2="46" stroke="currentColor" strokeWidth="1.2"/>
    <line x1="80" y1="12" x2="80" y2="46" stroke="currentColor" strokeWidth="1.2"/>
    <rect x="36" y="46" width="48" height="6" stroke="currentColor" strokeWidth="1.5" fill="none" rx="1"/>
    <line x1="4" y1="58" x2="116" y2="58" stroke="currentColor" strokeWidth="2"/>
  </svg>
);

// Usage
<BridgeLogo size={80} className="text-primary" />
```

---

## 2. Button Component

```jsx
export const Button = ({ 
  children, 
  variant = 'primary', 
  size = 'md', 
  icon: Icon,
  disabled = false,
  className = '',
  ...props 
}) => {
  const base = 'inline-flex items-center justify-center font-semibold transition-all duration-200 rounded-lg gap-2 disabled:opacity-50 disabled:cursor-not-allowed';
  
  const variants = {
    primary: 'bg-primary hover:bg-primary-dark text-white shadow-md hover:shadow-lg',
    secondary: 'bg-gray-100 dark:bg-gray-800 hover:bg-gray-200 dark:hover:bg-gray-700 text-gray-900 dark:text-white',
    outline: 'border-2 border-primary text-primary hover:bg-primary hover:text-white',
    ghost: 'text-primary hover:bg-primary/10',
    danger: 'bg-red-500 hover:bg-red-600 text-white',
  };
  
  const sizes = {
    sm: 'px-3 py-1.5 text-sm',
    md: 'px-5 py-2.5 text-base',
    lg: 'px-7 py-3.5 text-lg',
  };

  return (
    <button 
      className={`${base} ${variants[variant]} ${sizes[size]} ${className}`}
      disabled={disabled}
      {...props}
    >
      {Icon && <Icon size={size === 'sm' ? 16 : size === 'lg' ? 22 : 18} />}
      {children}
    </button>
  );
};

// Usage
<Button variant="primary" size="md">Get Started</Button>
<Button variant="outline" icon={ArrowRight}>Learn More</Button>
<Button variant="ghost" size="sm">Cancel</Button>
```

---

## 3. Input Component

```jsx
export const Input = ({ 
  label, 
  error, 
  icon: Icon, 
  className = '', 
  ...props 
}) => (
  <div className={className}>
    {label && (
      <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1.5">
        {label}
      </label>
    )}
    <div className="relative">
      {Icon && (
        <Icon 
          className="absolute left-3 top-1/2 -translate-y-1/2 text-gray-400" 
          size={18} 
        />
      )}
      <input
        className={`
          w-full px-4 py-2.5 rounded-lg border transition-colors
          ${Icon ? 'pl-10' : ''}
          ${error 
            ? 'border-red-500 focus:ring-red-500/50' 
            : 'border-gray-300 dark:border-gray-600 focus:border-primary focus:ring-primary/50'
          }
          bg-white dark:bg-gray-800 
          text-gray-900 dark:text-white 
          placeholder-gray-400
          focus:outline-none focus:ring-2
        `}
        {...props}
      />
    </div>
    {error && <p className="mt-1 text-sm text-red-500">{error}</p>}
  </div>
);

// Usage
<Input label="Email" type="email" placeholder="you@example.com" icon={Mail} />
<Input label="Password" type="password" error="Password is required" />
```

---

## 4. Select Component

```jsx
export const Select = ({ 
  label, 
  options, 
  className = '', 
  ...props 
}) => (
  <div className={className}>
    {label && (
      <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1.5">
        {label}
      </label>
    )}
    <select 
      className="
        w-full px-4 py-2.5 rounded-lg border 
        border-gray-300 dark:border-gray-600 
        bg-white dark:bg-gray-800 
        text-gray-900 dark:text-white 
        focus:outline-none focus:ring-2 focus:ring-primary/50 focus:border-primary
      " 
      {...props}
    >
      {options.map(opt => (
        <option key={opt.value} value={opt.value}>
          {opt.label}
        </option>
      ))}
    </select>
  </div>
);

// Usage
<Select 
  label="Service" 
  options={[
    { value: '', label: 'Select a service' },
    { value: 'assessment', label: 'AI Readiness Assessment' },
    { value: 'training', label: 'Team Training' },
    { value: 'implementation', label: 'Implementation' },
  ]} 
/>
```

---

## 5. Textarea Component

```jsx
export const Textarea = ({ 
  label, 
  error, 
  className = '', 
  ...props 
}) => (
  <div className={className}>
    {label && (
      <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1.5">
        {label}
      </label>
    )}
    <textarea
      className={`
        w-full px-4 py-2.5 rounded-lg border transition-colors resize-none
        ${error 
          ? 'border-red-500 focus:ring-red-500/50' 
          : 'border-gray-300 dark:border-gray-600 focus:border-primary focus:ring-primary/50'
        }
        bg-white dark:bg-gray-800 
        text-gray-900 dark:text-white 
        placeholder-gray-400
        focus:outline-none focus:ring-2
      `}
      {...props}
    />
    {error && <p className="mt-1 text-sm text-red-500">{error}</p>}
  </div>
);

// Usage
<Textarea label="Message" rows={4} placeholder="Tell us about your project..." />
```

---

## 6. Card Component

```jsx
export const Card = ({ 
  children, 
  variant = 'default', 
  className = '' 
}) => {
  const variants = {
    default: 'bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700',
    elevated: 'bg-white dark:bg-gray-800 shadow-lg border border-gray-100 dark:border-gray-700',
    primary: 'bg-primary text-white',
    cream: 'bg-cream dark:bg-gray-800 border border-gray-200 dark:border-gray-700',
  };

  return (
    <div className={`rounded-xl p-6 ${variants[variant]} ${className}`}>
      {children}
    </div>
  );
};

// Usage
<Card variant="elevated">
  <h3 className="font-semibold mb-2">Card Title</h3>
  <p className="text-gray-600">Card content goes here.</p>
</Card>
```

---

## 7. Badge Component

```jsx
export const Badge = ({ 
  children, 
  variant = 'default' 
}) => {
  const variants = {
    default: 'bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300',
    primary: 'bg-primary/10 text-primary',
    success: 'bg-green-100 dark:bg-green-900/30 text-green-700 dark:text-green-400',
    warning: 'bg-yellow-100 dark:bg-yellow-900/30 text-yellow-700 dark:text-yellow-400',
    danger: 'bg-red-100 dark:bg-red-900/30 text-red-700 dark:text-red-400',
  };

  return (
    <span className={`inline-flex px-2.5 py-0.5 rounded-full text-xs font-medium ${variants[variant]}`}>
      {children}
    </span>
  );
};

// Usage
<Badge variant="success">Active</Badge>
<Badge variant="warning">Pending</Badge>
<Badge variant="danger">Overdue</Badge>
```

---

## 8. Alert Component

```jsx
import { Info, CheckCircle, AlertTriangle, XCircle } from 'lucide-react';

export const Alert = ({ 
  children, 
  variant = 'info', 
  title,
  onClose 
}) => {
  const config = {
    info: {
      styles: 'bg-blue-50 dark:bg-blue-900/20 border-blue-200 dark:border-blue-800 text-blue-800 dark:text-blue-300',
      icon: Info,
    },
    success: {
      styles: 'bg-green-50 dark:bg-green-900/20 border-green-200 dark:border-green-800 text-green-800 dark:text-green-300',
      icon: CheckCircle,
    },
    warning: {
      styles: 'bg-yellow-50 dark:bg-yellow-900/20 border-yellow-200 dark:border-yellow-800 text-yellow-800 dark:text-yellow-300',
      icon: AlertTriangle,
    },
    danger: {
      styles: 'bg-red-50 dark:bg-red-900/20 border-red-200 dark:border-red-800 text-red-800 dark:text-red-300',
      icon: XCircle,
    },
  };

  const { styles, icon: Icon } = config[variant];

  return (
    <div className={`flex items-start gap-3 p-4 rounded-lg border ${styles}`}>
      <Icon size={20} className="flex-shrink-0 mt-0.5" />
      <div className="flex-1">
        {title && <div className="font-semibold mb-1">{title}</div>}
        <div>{children}</div>
      </div>
      {onClose && (
        <button onClick={onClose} className="opacity-70 hover:opacity-100">
          <X size={18} />
        </button>
      )}
    </div>
  );
};

// Usage
<Alert variant="success" title="Success!">Your changes have been saved.</Alert>
<Alert variant="danger">Something went wrong. Please try again.</Alert>
```

---

## 9. Icon Badge Component

```jsx
export const IconBadge = ({ 
  icon: Icon, 
  variant = 'light' 
}) => {
  const variants = {
    light: 'bg-primary/10 text-primary',
    solid: 'bg-primary text-white',
    outline: 'border-2 border-primary text-primary',
  };

  return (
    <div className={`w-12 h-12 rounded-xl flex items-center justify-center ${variants[variant]}`}>
      <Icon size={24} />
    </div>
  );
};

// Usage
<IconBadge icon={Shield} variant="light" />
<IconBadge icon={Award} variant="solid" />
```
