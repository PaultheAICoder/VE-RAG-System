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
    <nav className="sticky top-[45px] z-40 bg-[#F7F7F8] dark:bg-[#202123] border-b border-[#E5E5E5] dark:border-[#4E4F60]">
      <div className="max-w-7xl mx-auto px-4">
        <div className="flex gap-0.5 overflow-x-auto hide-scrollbar">
          {visibleItems.map((item) => {
            const Icon = item.icon;
            const isActive = location.pathname === item.href;

            return (
              <Link
                key={item.href}
                to={item.href}
                className={`
                  flex items-center gap-2 px-4 py-2.5 text-sm font-medium
                  border-b-2 -mb-px whitespace-nowrap transition-colors rounded-t-md
                  ${isActive
                    ? 'border-[#10A37F] text-[#10A37F] bg-white dark:bg-[#343541]'
                    : 'border-transparent text-[#6E6E80] dark:text-[#ACACBE] hover:text-[#2D2D2D] dark:hover:text-[#ECECF1] hover:bg-[#ECECF1] dark:hover:bg-[#2A2B32]'
                  }
                `}
              >
                <Icon size={16} />
                {item.label}
              </Link>
            );
          })}
        </div>
      </div>
    </nav>
  );
}
