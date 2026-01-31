# React UI Specification Context

**Created**: 2026-01-30
**Purpose**: Preserve context for React UI migration spec before compaction

## Decision Summary

| Category | Decision |
|----------|----------|
| Framework | React 18 + Vite + TypeScript |
| Styling | Tailwind CSS + AI Ready PDX design system |
| State Management | Zustand (auth/UI) + TanStack Query (API) |
| Deployment | Option B - Build on laptop, deploy via rsync, no npm on server |
| Layout | Top nav (not sidebar) |
| WebSocket | Yes - streaming chat responses |
| Dark Mode | Yes - toggle in header |
| Drag-drop Upload | Yes |
| Keyboard Shortcuts | Yes |
| Mobile Responsive | No (Rev 2) |
| Embeddable Widget | No (Rev 2) |

## Roles & Permissions

| View | System Admin | Customer Admin | User |
|------|:---:|:---:|:---:|
| Chat | ✓ | ✓ | ✓ |
| Documents | ✓ | ✓ | ✓ |
| Tags | ✓ | ✓ | ✗ |
| Users | ✓ | ✓ | ✗ |
| Settings | ✓ | ✗ | ✗ |
| Health Dashboard | ✓ | ✗ | ✗ |

- Customer Admin can create other Customer Admins
- Customer Admin has access to all tags

## Style Guide Location

`/home/jjob/project_data_files/VE-Rag-style-guides/`
- `aireadypdx-tokens-json.json` - Design tokens
- `aireadypdx-full-package.md` - Colors, typography, Tailwind config
- `aireadypdx-components.md` - Button, Input, Card, Badge, Alert components
- `aireadypdx-navigation.md` - Header, Sidebar, Tabs, Table components

## Key Design Tokens

- Primary: #2A9D8F (teal)
- Fonts: Poppins (headings), Inter (body)
- Dark mode: Full support with separate color palette

## Deployment Workflow

```bash
# On laptop
cd frontend
npm run build
rsync -avz dist/ spark:/srv/VE-RAG-System/frontend/dist/
```

## Layout Choice

Top nav layout with:
- Logo + App name left
- Nav items center (role-filtered)
- Dark mode toggle + User menu right
- Full-width content area below

## Views Required

1. **Chat** - Session list sidebar, message thread, streaming responses
2. **Documents** - Table with search/filter, bulk actions, upload modal
3. **Tags** - CRUD for tags (Admin only)
4. **Users** - User management with role assignment (Admin only)
5. **Settings** - Model config, processing options, query routing (System Admin only)
6. **Health** - RAG pipeline status, Vector DB stats, Ollama status (System Admin only)

## Rev 2 Items (Not in Spec)

- Mobile responsive design
- Embeddable chat widget for company intranet
- HR Help / IT Help standalone chat boxes

## Spec Location

`specs/react-ui-v1.md` in repo
