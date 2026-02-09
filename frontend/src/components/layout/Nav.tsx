import { Link, useLocation } from 'react-router-dom';
import {
  MessageSquare,
  FileText,
  Tags,
  Users,
  Settings,
  Activity,
  Sparkles,
} from 'lucide-react';
import { useAuthStore } from '../../stores/authStore';
import type { NavItem, UserRole } from '../../types';

const navItems: NavItem[] = [
  { href: '/chat', label: 'Chat', icon: MessageSquare, roles: ['admin', 'customer_admin', 'user'] },
  { href: '/documents', label: 'Documents', icon: FileText, roles: ['admin', 'customer_admin', 'user'] },
  { href: '/tags', label: 'Tags', icon: Tags, roles: ['admin', 'customer_admin'] },
  { href: '/users', label: 'Users', icon: Users, roles: ['admin', 'customer_admin'] },
  { href: '/rag-quality', label: 'RAG Quality', icon: Sparkles, roles: ['admin'] },
  { href: '/settings', label: 'Settings', icon: Settings, roles: ['admin'] },
  { href: '/health', label: 'Health', icon: Activity, roles: ['admin'] },
];

export function Nav() {
  const location = useLocation();
  const { user } = useAuthStore();
  const userRole = user?.role as UserRole;

  const visibleItems = navItems.filter((item) =>
    item.roles.includes(userRole)
  );

  return (
    <nav className="sticky top-14 z-40 border-b border-gray-100 dark:border-[#1E1E22] bg-white/80 dark:bg-[#0A0A0B]/80 glass">
      <div className="max-w-7xl mx-auto px-6">
        <div className="flex gap-0.5 overflow-x-auto hide-scrollbar">
          {visibleItems.map((item) => {
            const Icon = item.icon;
            const isActive = location.pathname === item.href;

            return (
              <Link
                key={item.href}
                to={item.href}
                className={`
                  relative flex items-center gap-2 px-4 py-3 text-[13px] font-medium
                  whitespace-nowrap transition-all duration-150
                  ${isActive
                    ? 'text-primary dark:text-primary-light'
                    : 'text-gray-500 dark:text-gray-500 hover:text-gray-900 dark:hover:text-gray-300'
                  }
                `}
              >
                <Icon size={16} />
                {item.label}
                {isActive && (
                  <span className="absolute bottom-0 left-4 right-4 h-[2px] bg-primary dark:bg-primary-light rounded-full" />
                )}
              </Link>
            );
          })}
        </div>
      </div>
    </nav>
  );
}
