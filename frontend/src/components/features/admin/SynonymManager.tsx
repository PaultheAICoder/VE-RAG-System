import { Card } from '../../ui';

export function SynonymManager() {
  return (
    <Card>
      <div className="text-center py-12">
        <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-2">
          Synonym Management
        </h3>
        <p className="text-gray-500 dark:text-gray-400">
          Synonym management coming soon. This feature will allow you to create
          term mappings (e.g., vacation → PTO) to improve search quality.
        </p>
      </div>
    </Card>
  );
}
