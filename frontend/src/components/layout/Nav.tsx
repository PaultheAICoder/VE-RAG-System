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
    <nav className="sticky top-[49px] z-40 corp-nav-gradient">
      <div className="max-w-7xl mx-auto px-4">
        <div className="flex gap-1 py-1 overflow-x-auto hide-scrollbar">
          {visibleItems.map((item) => {
            const Icon = item.icon;
            const isActive = location.pathname === item.href;

            return (
              <Link
                key={item.href}
                to={item.href}
                className={`
                  flex items-center gap-1.5 px-3 py-1.5 text-xs font-bold font-heading
                  whitespace-nowrap transition-colors
                  ${isActive
                    ? 'corp-button-active'
                    : 'corp-button text-gray-700 dark:text-gray-300'
                  }
                `}
              >
                <Icon size={14} />
                {item.label}
              </Link>
            );
          })}
        </div>
      </div>
    </nav>
  );
}
