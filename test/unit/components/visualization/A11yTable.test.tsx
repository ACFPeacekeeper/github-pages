import React from 'react';
import { describe, it, expect, vi } from 'vitest';
import { render, screen } from '@testing-library/react';
import { A11yTable } from '../../../../src/components/visualization/A11yTable';

describe('A11yTable', () => {
  const mockData = [
    { id: 1, name: 'Item 1', value: 10 },
    { id: 2, name: 'Item 2', value: 20 },
  ];

  const columns = [
    { key: 'name', header: 'Name' },
    { key: 'value', header: 'Value' },
  ];

  it('renders a table with caption and headers', () => {
    render(<A11yTable id="test-table" caption="Test Caption" data={mockData} columns={columns} />);
    expect(screen.getByText('Test Caption')).toBeInTheDocument();
    expect(screen.getByText('Name')).toBeInTheDocument();
    expect(screen.getByText('Value')).toBeInTheDocument();
  });

  it('renders data rows correctly', () => {
    render(<A11yTable id="test-table" caption="Test Caption" data={mockData} columns={columns} />);
    expect(screen.getByText('Item 1')).toBeInTheDocument();
    expect(screen.getByText('10')).toBeInTheDocument();
    expect(screen.getByText('Item 2')).toBeInTheDocument();
    expect(screen.getByText('20')).toBeInTheDocument();
  });

  it('handles empty data', () => {
    render(<A11yTable id="test-table" caption="Empty Table" data={[]} columns={columns} />);
    expect(screen.getByText('Empty Table')).toBeInTheDocument();
    expect(screen.getByText('No data available to display.')).toBeInTheDocument();
  });
});
