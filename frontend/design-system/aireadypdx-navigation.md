# AI Ready PDX — Navigation & Table Components

## 1. Header Navigation

```jsx
import { Menu, X } from 'lucide-react';
import { useState } from 'react';
import { BridgeLogo, Button } from './components';

export const Header = ({ 
  links = [], 
  currentPath = '/' 
}) => {
  const [mobileOpen, setMobileOpen] = useState(false);

  return (
    <header className="sticky top-0 z-50 bg-white/95 dark:bg-gray-900/95 backdrop-blur border-b border-gray-200 dark:border-gray-800">
      <div className="max-w-6xl mx-auto px-4 py-3 flex items-center justify-between">
        {/* Logo */}
        <a href="/" className="flex items-center gap-2">
          <span className="text-primary">
            <BridgeLogo size={48} />
          </span>
          <span className="font-bold text-sm text-gray-900 dark:text-white tracking-wide">
            AI READY PDX
          </span>
        </a>

        {/* Desktop Navigation */}
        <nav className="hidden md:flex items-center gap-1">
          {links.map(link => (
            <a
              key={link.href}
              href={link.href}
              className={`
                px-3 py-2 text-sm font-medium rounded-lg transition-colors
                ${currentPath === link.href
                  ? 'text-primary bg-primary/5'
                  : 'text-gray-600 dark:text-gray-300 hover:text-primary hover:bg-primary/5'
                }
              `}
            >
              {link.label}
            </a>
          ))}
        </nav>

        {/* CTA Button */}
        <div className="hidden md:block">
          <Button size="sm">Contact Us</Button>
        </div>

        {/* Mobile Menu Button */}
        <button
          className="md:hidden p-2 text-gray-600 dark:text-gray-300"
          onClick={() => setMobileOpen(!mobileOpen)}
        >
          {mobileOpen ? <X size={24} /> : <Menu size={24} />}
        </button>
      </div>

      {/* Mobile Navigation */}
      {mobileOpen && (
        <div className="md:hidden border-t border-gray-200 dark:border-gray-800 bg-white dark:bg-gray-900 px-4 py-4">
          <nav className="flex flex-col gap-1">
            {links.map(link => (
              <a
                key={link.href}
                href={link.href}
                className={`
                  px-3 py-2 text-sm font-medium rounded-lg
                  ${currentPath === link.href
                    ? 'text-primary bg-primary/5'
                    : 'text-gray-600 dark:text-gray-300'
                  }
                `}
              >
                {link.label}
              </a>
            ))}
            <Button className="mt-4 w-full">Contact Us</Button>
          </nav>
        </div>
      )}
    </header>
  );
};

// Usage
<Header 
  links={[
    { href: '/', label: 'Home' },
    { href: '/services', label: 'Services' },
    { href: '/pricing', label: 'Pricing' },
    { href: '/about', label: 'About' },
  ]}
  currentPath="/services"
/>
```

---

## 2. Sidebar Navigation

```jsx
import { Home, Users, Calendar, Settings, LogOut, ChevronLeft } from 'lucide-react';
import { BridgeLogo } from './components';

export const Sidebar = ({ 
  items = [], 
  currentPath = '/',
  collapsed = false,
  onToggle,
  user 
}) => {
  return (
    <aside className={`
      h-screen sticky top-0 flex flex-col
      bg-white dark:bg-gray-900 
      border-r border-gray-200 dark:border-gray-800
      transition-all duration-300
      ${collapsed ? 'w-16' : 'w-64'}
    `}>
      {/* Header */}
      <div className="p-4 border-b border-gray-200 dark:border-gray-800 flex items-center justify-between">
        <div className={`flex items-center gap-2 ${collapsed ? 'justify-center w-full' : ''}`}>
          <span className="text-primary">
            <BridgeLogo size={collapsed ? 32 : 40} />
          </span>
          {!collapsed && (
            <span className="font-bold text-sm text-gray-900 dark:text-white">
              AI READY PDX
            </span>
          )}
        </div>
        {onToggle && !collapsed && (
          <button 
            onClick={onToggle}
            className="p-1 rounded hover:bg-gray-100 dark:hover:bg-gray-800 text-gray-400"
          >
            <ChevronLeft size={18} />
          </button>
        )}
      </div>

      {/* Navigation Items */}
      <nav className="flex-1 p-3 space-y-1">
        {items.map(item => {
          const Icon = item.icon;
          const isActive = currentPath === item.href;
          
          return (
            <a
              key={item.href}
              href={item.href}
              className={`
                flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm font-medium
                transition-colors
                ${isActive
                  ? 'bg-primary text-white'
                  : 'text-gray-600 dark:text-gray-400 hover:bg-gray-100 dark:hover:bg-gray-800 hover:text-gray-900 dark:hover:text-white'
                }
                ${collapsed ? 'justify-center' : ''}
              `}
              title={collapsed ? item.label : undefined}
            >
              <Icon size={20} />
              {!collapsed && <span>{item.label}</span>}
            </a>
          );
        })}
      </nav>

      {/* User Section */}
      {user && (
        <div className={`p-3 border-t border-gray-200 dark:border-gray-800 ${collapsed ? 'text-center' : ''}`}>
          {!collapsed && (
            <div className="px-3 py-2 mb-2">
              <div className="font-medium text-sm text-gray-900 dark:text-white">{user.name}</div>
              <div className="text-xs text-gray-500">{user.email}</div>
            </div>
          )}
          <a
            href="/logout"
            className={`
              flex items-center gap-3 px-3 py-2 rounded-lg text-sm
              text-gray-600 dark:text-gray-400 
              hover:bg-gray-100 dark:hover:bg-gray-800
              ${collapsed ? 'justify-center' : ''}
            `}
          >
            <LogOut size={18} />
            {!collapsed && <span>Sign Out</span>}
          </a>
        </div>
      )}
    </aside>
  );
};

// Usage
<Sidebar
  items={[
    { href: '/dashboard', label: 'Dashboard', icon: Home },
    { href: '/clients', label: 'Clients', icon: Users },
    { href: '/schedule', label: 'Schedule', icon: Calendar },
    { href: '/settings', label: 'Settings', icon: Settings },
  ]}
  currentPath="/dashboard"
  user={{ name: 'John Doe', email: 'john@example.com' }}
/>
```

---

## 3. Tabs Component

```jsx
import { useState } from 'react';

export const Tabs = ({ 
  tabs = [], 
  defaultTab,
  onChange 
}) => {
  const [activeTab, setActiveTab] = useState(defaultTab || tabs[0]?.id);

  const handleChange = (id) => {
    setActiveTab(id);
    onChange?.(id);
  };

  return (
    <div>
      {/* Tab Headers */}
      <div className="border-b border-gray-200 dark:border-gray-700 flex gap-1 overflow-x-auto">
        {tabs.map(tab => (
          <button
            key={tab.id}
            onClick={() => handleChange(tab.id)}
            className={`
              px-4 py-2.5 text-sm font-medium border-b-2 -mb-px whitespace-nowrap
              transition-colors
              ${activeTab === tab.id
                ? 'border-primary text-primary'
                : 'border-transparent text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-300'
              }
            `}
          >
            {tab.label}
          </button>
        ))}
      </div>

      {/* Tab Content */}
      <div className="py-4">
        {tabs.find(t => t.id === activeTab)?.content}
      </div>
    </div>
  );
};

// Usage
<Tabs
  tabs={[
    { id: 'overview', label: 'Overview', content: <OverviewPanel /> },
    { id: 'analytics', label: 'Analytics', content: <AnalyticsPanel /> },
    { id: 'settings', label: 'Settings', content: <SettingsPanel /> },
  ]}
  defaultTab="overview"
/>
```

---

## 4. Breadcrumbs Component

```jsx
import { ChevronRight, Home } from 'lucide-react';

export const Breadcrumbs = ({ items = [] }) => {
  return (
    <nav className="flex items-center gap-2 text-sm">
      <a 
        href="/" 
        className="text-gray-400 hover:text-primary transition-colors"
      >
        <Home size={16} />
      </a>
      
      {items.map((item, index) => (
        <div key={item.href || index} className="flex items-center gap-2">
          <ChevronRight size={16} className="text-gray-300 dark:text-gray-600" />
          {item.href ? (
            <a 
              href={item.href}
              className="text-primary hover:underline"
            >
              {item.label}
            </a>
          ) : (
            <span className="text-gray-500 dark:text-gray-400">
              {item.label}
            </span>
          )}
        </div>
      ))}
    </nav>
  );
};

// Usage
<Breadcrumbs
  items={[
    { label: 'Services', href: '/services' },
    { label: 'AI Readiness Assessment' },
  ]}
/>
```

---

## 5. Pagination Component

```jsx
import { ChevronLeft, ChevronRight } from 'lucide-react';

export const Pagination = ({ 
  currentPage = 1, 
  totalPages = 10,
  onPageChange 
}) => {
  const getPageNumbers = () => {
    const pages = [];
    const showEllipsisStart = currentPage > 3;
    const showEllipsisEnd = currentPage < totalPages - 2;

    if (totalPages <= 7) {
      for (let i = 1; i <= totalPages; i++) pages.push(i);
    } else {
      pages.push(1);
      if (showEllipsisStart) pages.push('...');
      
      const start = Math.max(2, currentPage - 1);
      const end = Math.min(totalPages - 1, currentPage + 1);
      for (let i = start; i <= end; i++) pages.push(i);
      
      if (showEllipsisEnd) pages.push('...');
      pages.push(totalPages);
    }
    return pages;
  };

  return (
    <div className="flex items-center gap-1">
      <button
        onClick={() => onPageChange?.(currentPage - 1)}
        disabled={currentPage === 1}
        className="p-2 rounded-lg text-gray-500 hover:bg-gray-100 dark:hover:bg-gray-800 disabled:opacity-50 disabled:cursor-not-allowed"
      >
        <ChevronLeft size={18} />
      </button>

      {getPageNumbers().map((page, index) => (
        <button
          key={index}
          onClick={() => typeof page === 'number' && onPageChange?.(page)}
          disabled={page === '...'}
          className={`
            w-9 h-9 rounded-lg text-sm font-medium transition-colors
            ${page === currentPage
              ? 'bg-primary text-white'
              : page === '...'
                ? 'cursor-default text-gray-400'
                : 'text-gray-500 dark:text-gray-400 hover:bg-gray-100 dark:hover:bg-gray-800'
            }
          `}
        >
          {page}
        </button>
      ))}

      <button
        onClick={() => onPageChange?.(currentPage + 1)}
        disabled={currentPage === totalPages}
        className="p-2 rounded-lg text-gray-500 hover:bg-gray-100 dark:hover:bg-gray-800 disabled:opacity-50 disabled:cursor-not-allowed"
      >
        <ChevronRight size={18} />
      </button>
    </div>
  );
};

// Usage
<Pagination currentPage={3} totalPages={10} onPageChange={setPage} />
```

---

## 6. Table Component

```jsx
export const Table = ({ 
  columns = [], 
  data = [],
  onRowClick,
  className = '' 
}) => {
  return (
    <div className={`overflow-x-auto rounded-xl border border-gray-200 dark:border-gray-700 ${className}`}>
      <table className="w-full">
        <thead className="bg-gray-50 dark:bg-gray-800">
          <tr>
            {columns.map(col => (
              <th 
                key={col.key} 
                className="px-4 py-3 text-left text-sm font-semibold text-gray-600 dark:text-gray-300"
              >
                {col.label}
              </th>
            ))}
          </tr>
        </thead>
        <tbody className="divide-y divide-gray-200 dark:divide-gray-700 bg-white dark:bg-gray-900">
          {data.map((row, rowIndex) => (
            <tr 
              key={rowIndex} 
              onClick={() => onRowClick?.(row)}
              className={`
                hover:bg-gray-50 dark:hover:bg-gray-800 transition-colors
                ${onRowClick ? 'cursor-pointer' : ''}
              `}
            >
              {columns.map(col => (
                <td 
                  key={col.key} 
                  className="px-4 py-3 text-sm text-gray-600 dark:text-gray-400"
                >
                  {col.render ? col.render(row[col.key], row) : row[col.key]}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
};

// Usage
<Table
  columns={[
    { key: 'service', label: 'Service' },
    { 
      key: 'status', 
      label: 'Status',
      render: (value) => <Badge variant={value === 'Active' ? 'success' : 'warning'}>{value}</Badge>
    },
    { key: 'clients', label: 'Clients' },
    { key: 'revenue', label: 'Revenue' },
  ]}
  data={[
    { service: 'AI Readiness Assessment', status: 'Active', clients: '24', revenue: '$48,000' },
    { service: 'Team Training', status: 'Active', clients: '18', revenue: '$36,000' },
    { service: 'Implementation', status: 'Pending', clients: '6', revenue: '$72,000' },
  ]}
/>
```

---

## 7. Data Table with Sorting & Search

```jsx
import { useState, useMemo } from 'react';
import { Search, ChevronUp, ChevronDown } from 'lucide-react';

export const DataTable = ({ 
  columns = [], 
  data = [],
  searchable = true,
  searchPlaceholder = 'Search...'
}) => {
  const [search, setSearch] = useState('');
  const [sortKey, setSortKey] = useState(null);
  const [sortDir, setSortDir] = useState('asc');

  const handleSort = (key) => {
    if (sortKey === key) {
      setSortDir(sortDir === 'asc' ? 'desc' : 'asc');
    } else {
      setSortKey(key);
      setSortDir('asc');
    }
  };

  const filteredData = useMemo(() => {
    let result = [...data];
    
    // Filter
    if (search) {
      const term = search.toLowerCase();
      result = result.filter(row => 
        columns.some(col => 
          String(row[col.key]).toLowerCase().includes(term)
        )
      );
    }
    
    // Sort
    if (sortKey) {
      result.sort((a, b) => {
        const aVal = a[sortKey];
        const bVal = b[sortKey];
        const comparison = aVal < bVal ? -1 : aVal > bVal ? 1 : 0;
        return sortDir === 'asc' ? comparison : -comparison;
      });
    }
    
    return result;
  }, [data, search, sortKey, sortDir, columns]);

  return (
    <div>
      {/* Search */}
      {searchable && (
        <div className="mb-4 relative">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 text-gray-400" size={18} />
          <input
            type="text"
            placeholder={searchPlaceholder}
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            className="w-full max-w-xs pl-10 pr-4 py-2 rounded-lg border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-800 text-sm"
          />
        </div>
      )}

      {/* Table */}
      <div className="overflow-x-auto rounded-xl border border-gray-200 dark:border-gray-700">
        <table className="w-full">
          <thead className="bg-gray-50 dark:bg-gray-800">
            <tr>
              {columns.map(col => (
                <th 
                  key={col.key}
                  onClick={() => col.sortable !== false && handleSort(col.key)}
                  className={`
                    px-4 py-3 text-left text-sm font-semibold text-gray-600 dark:text-gray-300
                    ${col.sortable !== false ? 'cursor-pointer hover:bg-gray-100 dark:hover:bg-gray-700' : ''}
                  `}
                >
                  <div className="flex items-center gap-1">
                    {col.label}
                    {sortKey === col.key && (
                      sortDir === 'asc' ? <ChevronUp size={14} /> : <ChevronDown size={14} />
                    )}
                  </div>
                </th>
              ))}
            </tr>
          </thead>
          <tbody className="divide-y divide-gray-200 dark:divide-gray-700 bg-white dark:bg-gray-900">
            {filteredData.map((row, i) => (
              <tr key={i} className="hover:bg-gray-50 dark:hover:bg-gray-800">
                {columns.map(col => (
                  <td key={col.key} className="px-4 py-3 text-sm text-gray-600 dark:text-gray-400">
                    {col.render ? col.render(row[col.key], row) : row[col.key]}
                  </td>
                ))}
              </tr>
            ))}
            {filteredData.length === 0 && (
              <tr>
                <td colSpan={columns.length} className="px-4 py-8 text-center text-gray-500">
                  No results found
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
};
```
