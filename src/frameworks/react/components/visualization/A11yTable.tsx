import React from 'react';

interface Column<T> {
  key: keyof T | string;
  header: string;
  render?: (item: T) => React.ReactNode;
}

interface A11yTableProps<T> {
  id: string;
  caption: string;
  data: T[];
  columns: Column<T>[];
  className?: string;
}

export function A11yTable<T>({ id, caption, data, columns, className = 'sr-only' }: A11yTableProps<T>) {
  if (!data || data.length === 0) {
    return (
      <div id={id} className={className}>
        <p>{caption}</p>
        <p>No data available to display.</p>
      </div>
    );
  }

  return (
    <table id={id} className={className}>
      <caption>{caption}</caption>
      <thead>
        <tr>
          {columns.map((col) => (
            <th key={String(col.key)} scope="col">
              {col.header}
            </th>
          ))}
        </tr>
      </thead>
      <tbody>
        {data.map((item, index) => (
          <tr key={index}>
            {columns.map((col) => (
              <td key={String(col.key)}>
                {col.render ? col.render(item) : String((item as any)[col.key])}
              </td>
            ))}
          </tr>
        ))}
      </tbody>
    </table>
  );
}
