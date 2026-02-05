import { Card } from '../../ui';

export function QAManager() {
  return (
    <Card>
      <div className="text-center py-12">
        <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-2">
          Curated Q&A Management
        </h3>
        <p className="text-gray-500 dark:text-gray-400">
          Curated Q&A management coming soon. This feature will allow you to create
          admin-approved answers for specific questions that bypass the RAG pipeline.
        </p>
      </div>
    </Card>
  );
}
